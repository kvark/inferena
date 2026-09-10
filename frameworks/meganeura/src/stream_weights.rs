//! Experiment-only single-file reader: keep one tensor, not the whole checkpoint.

use meganeura::data::safetensors::SafeTensorsModel;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct StreamWeights {
    file: File,
    tensors: BTreeMap<String, Tensor>,
    data_start: u64,
}

impl StreamWeights {
    fn open(path: PathBuf) -> Self {
        let mut file = File::open(path).expect("checkpoint file");
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix).unwrap();
        let header_size = u64::from_le_bytes(prefix);
        assert!(
            header_size <= 8 * 1024 * 1024,
            "checkpoint header exceeds diagnostic cap"
        );
        let mut header = vec![0; header_size as usize];
        file.read_exact(&mut header).unwrap();
        assert_eq!(header.first(), Some(&b'{'));
        let mut values: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header).unwrap();
        if let Some(metadata) = values.remove("__metadata__") {
            let _: BTreeMap<String, String> =
                serde_json::from_value(metadata).expect("string metadata");
        }
        let tensors: BTreeMap<String, Tensor> = values
            .into_iter()
            .map(|(name, value)| {
                (
                    name,
                    serde_json::from_value(value).expect("tensor metadata"),
                )
            })
            .collect();
        let mut ordered: Vec<_> = tensors.values().collect();
        ordered.sort_by_key(|tensor| tensor.data_offsets);
        let mut end = 0usize;
        for tensor in ordered {
            assert_eq!(
                tensor.data_offsets[0], end,
                "checkpoint offsets must be contiguous"
            );
            let width = match tensor.dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                other => panic!("unsupported floating checkpoint dtype: {other}"),
            };
            let elements = tensor
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .unwrap();
            let bytes = elements.checked_mul(width).unwrap();
            end = end.checked_add(bytes).unwrap();
            assert_eq!(
                tensor.data_offsets[1], end,
                "checkpoint shape/size mismatch"
            );
            assert!(
                bytes <= 512 * 1024 * 1024,
                "stored tensor exceeds diagnostic cap"
            );
        }
        let data_start = 8 + header_size;
        assert_eq!(
            data_start.checked_add(end as u64).unwrap(),
            file.metadata().unwrap().len(),
            "checkpoint data length mismatch"
        );
        Self {
            file,
            tensors,
            data_start,
        }
    }

    fn tensor(&mut self, name: &str) -> SafeTensorsModel {
        let tensor = self.tensors.get(name).expect("checkpoint tensor");
        let bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
        let mut local = tensor.clone();
        local.data_offsets = [0, bytes];
        let mut header = serde_json::to_vec(&BTreeMap::from([(name, local)])).unwrap();
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let start = 8 + header.len();
        let mut buffer = Vec::with_capacity(start + bytes);
        buffer.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&header);
        buffer.resize(start + bytes, 0);
        self.file
            .seek(SeekFrom::Start(
                self.data_start + tensor.data_offsets[0] as u64,
            ))
            .unwrap();
        self.file.read_exact(&mut buffer[start..]).unwrap();
        SafeTensorsModel::from_bytes(buffer).expect("validated one-tensor checkpoint")
    }
}

pub enum Weights {
    Resident(SafeTensorsModel),
    Stream(StreamWeights),
}

impl Weights {
    pub fn open(path: PathBuf, stream: bool) -> Self {
        if stream {
            Self::Stream(StreamWeights::open(path))
        } else {
            Self::Resident(SafeTensorsModel::load(path).expect("local model load failed"))
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        match self {
            Self::Resident(model) => model.tensor_info().contains_key(name),
            Self::Stream(model) => model.tensors.contains_key(name),
        }
    }

    pub fn tensor_f32(&mut self, name: &str, transposed: bool) -> Vec<f32> {
        let local;
        let model: &SafeTensorsModel = match self {
            Self::Resident(model) => model,
            Self::Stream(model) => {
                local = model.tensor(name);
                &local
            }
        };
        if transposed {
            model.tensor_f32_auto_transposed(name).unwrap()
        } else {
            model.tensor_f32_auto(name).unwrap()
        }
    }
}
