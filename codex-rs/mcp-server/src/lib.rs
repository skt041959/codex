//! Prototype MCP server.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use std::io::ErrorKind;
use std::io::Result as IoResult;
use std::net::SocketAddr;
use std::sync::Arc;

use codex_arg0::Arg0DispatchPaths;
use codex_core::AuthManager;
use codex_core::ThreadManager;
use codex_core::config::Config;
use codex_protocol::protocol::SessionSource;
use codex_utils_cli::CliConfigOverrides;

use futures::SinkExt;
use futures::StreamExt;
use rmcp::model::ClientNotification;
use rmcp::model::ClientRequest;
use rmcp::model::JsonRpcMessage;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::{self};
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;
use tracing_subscriber::EnvFilter;

mod codex_tool_config;
mod codex_tool_runner;
mod exec_approval;
pub(crate) mod message_processor;
mod outgoing_message;
mod patch_approval;

use crate::message_processor::MessageProcessor;
use crate::outgoing_message::OutgoingJsonRpcMessage;
use crate::outgoing_message::OutgoingMessage;
use crate::outgoing_message::OutgoingMessageSender;

pub use crate::codex_tool_config::CodexToolCallParam;
pub use crate::codex_tool_config::CodexToolCallReplyParam;
pub use crate::exec_approval::ExecApprovalElicitRequestParams;
pub use crate::exec_approval::ExecApprovalResponse;
pub use crate::patch_approval::PatchApprovalElicitRequestParams;
pub use crate::patch_approval::PatchApprovalResponse;

/// Size of the bounded channels used to communicate between tasks. The value
/// is a balance between throughput and memory usage – 128 messages should be
/// plenty for an interactive CLI.
const CHANNEL_CAPACITY: usize = 128;

type IncomingMessage = JsonRpcMessage<ClientRequest, Value, ClientNotification>;

pub async fn run_main(
    arg0_paths: Arg0DispatchPaths,
    cli_config_overrides: CliConfigOverrides,
) -> IoResult<()> {
    // Install a simple subscriber so `tracing` output is visible.  Users can
    // control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse CLI overrides once and derive the base Config eagerly so later
    // components do not need to work with raw TOML values.
    let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
        std::io::Error::new(
            ErrorKind::InvalidInput,
            format!("error parsing -c overrides: {e}"),
        )
    })?;
    let config = Config::load_with_cli_overrides(cli_kv_overrides)
        .await
        .map_err(|e| {
            std::io::Error::new(ErrorKind::InvalidData, format!("error loading config: {e}"))
        })?;
    let config = Arc::new(config);

    let auth_manager = AuthManager::shared(
        config.codex_home.clone(),
        false,
        config.cli_auth_credentials_store_mode,
    );
    let thread_manager = Arc::new(ThreadManager::new(
        config.codex_home.clone(),
        auth_manager,
        SessionSource::Mcp,
        config.model_catalog.clone(),
    ));

    // Set up channels.
    let (incoming_tx, mut incoming_rx) = mpsc::channel::<IncomingMessage>(CHANNEL_CAPACITY);
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<OutgoingMessage>();

    // Task: read from stdin, push to `incoming_tx`.
    let stdin_reader_handle = tokio::spawn({
        async move {
            let stdin = io::stdin();
            let reader = BufReader::new(stdin);
            let mut lines = reader.lines();

            while let Some(line) = lines.next_line().await.unwrap_or_default() {
                match serde_json::from_str::<IncomingMessage>(&line) {
                    Ok(msg) => {
                        if incoming_tx.send(msg).await.is_err() {
                            // Receiver gone – nothing left to do.
                            break;
                        }
                    }
                    Err(e) => error!("Failed to deserialize JSON-RPC message: {e}"),
                }
            }

            debug!("stdin reader finished (EOF)");
        }
    });

    // Task: process incoming messages.
    let processor_handle = tokio::spawn({
        let outgoing_message_sender = OutgoingMessageSender::new(outgoing_tx);
        let mut processor = MessageProcessor::new(
            outgoing_message_sender,
            arg0_paths,
            thread_manager,
        );
        async move {
            while let Some(msg) = incoming_rx.recv().await {
                match msg {
                    JsonRpcMessage::Request(r) => processor.process_request(r).await,
                    JsonRpcMessage::Response(r) => processor.process_response(r).await,
                    JsonRpcMessage::Notification(n) => processor.process_notification(n).await,
                    JsonRpcMessage::Error(e) => processor.process_error(e),
                }
            }

            info!("processor task exited (channel closed)");
        }
    });

    // Task: write outgoing messages to stdout.
    let stdout_writer_handle = tokio::spawn(async move {
        let mut stdout = io::stdout();
        while let Some(outgoing_message) = outgoing_rx.recv().await {
            let msg: OutgoingJsonRpcMessage = outgoing_message.into();
            match serde_json::to_string(&msg) {
                Ok(json) => {
                    if let Err(e) = stdout.write_all(json.as_bytes()).await {
                        error!("Failed to write to stdout: {e}");
                        break;
                    }
                    if let Err(e) = stdout.write_all(b"\n").await {
                        error!("Failed to write newline to stdout: {e}");
                        break;
                    }
                }
                Err(e) => error!("Failed to serialize JSON-RPC message: {e}"),
            }
        }

        info!("stdout writer exited (channel closed)");
    });

    // Wait for all tasks to finish.  The typical exit path is the stdin reader
    // hitting EOF which, once it drops `incoming_tx`, propagates shutdown to
    // the processor and then to the stdout task.
    let _ = tokio::join!(stdin_reader_handle, processor_handle, stdout_writer_handle);

    Ok(())
}

pub async fn run_network_server(
    port: u16,
    arg0_paths: Arg0DispatchPaths,
    thread_manager: Arc<ThreadManager>,
) -> IoResult<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;
    info!("MCP network server listening on ws://{addr}");

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        info!("Accepted connection from {peer_addr}");

        let arg0_paths = arg0_paths.clone();
        let thread_manager = thread_manager.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, arg0_paths, thread_manager).await {
                error!("Error handling connection from {peer_addr}: {e}");
            }
        });
    }
}

async fn handle_connection(
    stream: TcpStream,
    arg0_paths: Arg0DispatchPaths,
    thread_manager: Arc<ThreadManager>,
) -> IoResult<()> {
    let ws_stream = accept_async(stream).await.map_err(|e| {
        std::io::Error::new(
            ErrorKind::ConnectionAborted,
            format!("WebSocket handshake failed: {e}"),
        )
    })?;

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<OutgoingMessage>();
    let (incoming_tx, mut incoming_rx) = mpsc::channel::<IncomingMessage>(CHANNEL_CAPACITY);

    let sender_handle: JoinHandle<IoResult<()>> = tokio::spawn(async move {
        while let Some(msg) = outgoing_rx.recv().await {
            let json_rpc_msg: OutgoingJsonRpcMessage = msg.into();
            let json = serde_json::to_string(&json_rpc_msg).map_err(|e| {
                std::io::Error::new(ErrorKind::InvalidData, format!("Serialization error: {e}"))
            })?;
            ws_sender
                .send(TungsteniteMessage::Text(json.into()))
                .await
                .map_err(|e| {
                    std::io::Error::new(ErrorKind::ConnectionAborted, format!("Send error: {e}"))
                })?;
        }
        Ok(())
    });

    let processor_handle = tokio::spawn({
        let outgoing_sender = OutgoingMessageSender::new(outgoing_tx);
        let mut processor =
            MessageProcessor::new(outgoing_sender, arg0_paths, thread_manager.clone());
        async move {
            while let Some(msg) = incoming_rx.recv().await {
                match msg {
                    JsonRpcMessage::Request(r) => processor.process_request(r).await,
                    JsonRpcMessage::Response(r) => processor.process_response(r).await,
                    JsonRpcMessage::Notification(n) => processor.process_notification(n).await,
                    JsonRpcMessage::Error(e) => processor.process_error(e),
                }
            }
        }
    });

    while let Some(msg) = ws_receiver.next().await {
        let msg = msg.map_err(|e| {
            std::io::Error::new(ErrorKind::ConnectionAborted, format!("Receive error: {e}"))
        })?;

        if let TungsteniteMessage::Text(text) = msg {
            match serde_json::from_str::<IncomingMessage>(&text) {
                Ok(json_rpc_msg) => {
                    if incoming_tx.send(json_rpc_msg).await.is_err() {
                        break;
                    }
                }
                Err(e) => warn!("Failed to deserialize message: {e}"),
            }
        } else if msg.is_close() {
            break;
        }
    }

    let _ = sender_handle.await;
    let _ = processor_handle.await;

    Ok(())
}
