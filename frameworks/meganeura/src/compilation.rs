//! Experiment-only CPU span export. No GPU timestamps or scheduling changes.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{Subscriber, field, span};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

#[derive(Default)]
struct Fields(BTreeMap<String, String>);

impl field::Visit for Fields {
    fn record_debug(&mut self, field: &field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

struct SpanData {
    fields: Fields,
    entered: Option<Instant>,
}

struct Timings {
    epoch: Instant,
    output: Arc<Mutex<BufWriter<File>>>,
}

pub struct Capture(Arc<Mutex<BufWriter<File>>>);

impl Drop for Capture {
    fn drop(&mut self) {
        self.0.lock().unwrap().flush().unwrap();
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Timings {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        ctx.span(id).unwrap().extensions_mut().insert(SpanData {
            fields,
            entered: None,
        });
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        ctx.span(id)
            .unwrap()
            .extensions_mut()
            .get_mut::<SpanData>()
            .unwrap()
            .entered = Some(Instant::now());
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        let end = Instant::now();
        let current = ctx.span(id).unwrap();
        let ext = current.extensions();
        let data = ext.get::<SpanData>().unwrap();
        let start = data.entered.unwrap();
        let ancestors: Vec<_> = current.scope().skip(1).map(|ancestor| {
            let ext = ancestor.extensions();
            serde_json::json!({"name": ancestor.name(), "fields": ext.get::<SpanData>().map(|data| &data.fields.0)})
        }).collect();
        let row = serde_json::json!({
            "stage": current.name(), "start_ns": start.duration_since(self.epoch).as_nanos(),
            "duration_ns": end.duration_since(start).as_nanos(), "fields": data.fields.0,
            "ancestors": ancestors, "thread": format!("{:?}", std::thread::current().id()),
        });
        let mut writer = self.output.lock().unwrap();
        serde_json::to_writer(&mut *writer, &row).unwrap();
        writeln!(writer).unwrap();
    }
}

pub fn init() -> Option<Capture> {
    let path = std::env::var_os("INFERENA_COMPILE_TRACE")?;
    let file = File::create_new(path).expect("compilation trace must be a new file");
    let output = Arc::new(Mutex::new(BufWriter::new(file)));
    tracing_subscriber::registry()
        .with(Timings {
            epoch: Instant::now(),
            output: output.clone(),
        })
        .init();
    Some(Capture(output))
}
