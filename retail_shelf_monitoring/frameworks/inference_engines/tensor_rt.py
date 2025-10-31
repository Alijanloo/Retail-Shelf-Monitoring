import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from retail_shelf_monitoring.usecases.interfaces.inference_model import InferenceModel


class TensorRTModel(InferenceModel):
    def __init__(
        self,
        onnx_path: str = None,
        engine_path: str = None,
        max_batch_size: int = 8,
        workspace_size: int = 1 << 30,
    ):
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.context = None

        if engine_path:
            self.engine = self._load_engine(engine_path)
        elif onnx_path:
            self.engine = self._build_engine(onnx_path, max_batch_size, workspace_size)
        else:
            raise ValueError(
                "You must provide either an ONNX file path or an engine file path."
            )

        self.context = self.engine.create_execution_context()
        print("TensorRTModel initialized successfully.")

    @property
    def input_shape(self):
        """Return the input shape of the model (excluding batch dimension)."""
        return tuple(self.engine.get_binding_shape(0)[1:])

    def _build_engine(self, onnx_file_path, max_batch_size=8, workspace_size=1 << 30):
        """Build a TensorRT engine from ONNX model"""
        with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, trt.OnnxParser(network, self.TRT_LOGGER) as parser:
            builder.max_batch_size = max_batch_size
            builder.max_workspace_size = workspace_size

            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("Failed to parse ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("ONNX parsing failed.")

            print("Building TensorRT engine...")
            engine = builder.build_cuda_engine(network)
            print("Engine built successfully.")
            return engine

    def save_engine(self, path: str):
        """Serialize and save the TensorRT engine."""
        with open(path, "wb") as f:
            f.write(self.engine.serialize())
        print(f"Engine saved to {path}")

    def _load_engine(self, engine_file_path):
        """Load a serialized TensorRT engine"""
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            print(f"Loading TensorRT engine from {engine_file_path}...")
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, batch_size=1):
        h_inputs, h_outputs = [], []
        d_inputs, d_outputs = [], []
        bindings = []

        for binding in self.engine:
            shape = list(self.engine.get_binding_shape(binding))
            shape[0] = batch_size
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                h_inputs.append(host_mem)
                d_inputs.append(device_mem)
            else:
                h_outputs.append(host_mem)
                d_outputs.append(device_mem)

        return h_inputs, d_inputs, h_outputs, d_outputs, bindings

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        batch_size = input_data.shape[0]
        h_inputs, d_inputs, h_outputs, d_outputs, bindings = self._allocate_buffers(
            batch_size
        )

        np.copyto(h_inputs[0], input_data.ravel())
        cuda.memcpy_htod(d_inputs[0], h_inputs[0])

        self.context.execute_v2(bindings)

        cuda.memcpy_dtoh(h_outputs[0], d_outputs[0])
        output_shape = [batch_size] + list(self.engine.get_binding_shape(1))[1:]
        output = h_outputs[0].reshape(output_shape)
        return output

    def batch_infer(self, data_loader, batch_size=8) -> np.ndarray:
        results = []
        for i in range(0, len(data_loader), batch_size):
            batch = np.array(data_loader[i : i + batch_size])
            actual_batch_size = batch.shape[0]

            h_inputs, d_inputs, h_outputs, d_outputs, bindings = self._allocate_buffers(
                actual_batch_size
            )

            np.copyto(h_inputs[0], batch.ravel())
            cuda.memcpy_htod(d_inputs[0], h_inputs[0])

            self.context.execute_v2(bindings)

            cuda.memcpy_dtoh(h_outputs[0], d_outputs[0])
            output_shape = [actual_batch_size] + list(self.engine.get_binding_shape(1))[
                1:
            ]
            batch_output = h_outputs[0].reshape(output_shape)
            results.append(batch_output)

        return np.concatenate(results, axis=0)
