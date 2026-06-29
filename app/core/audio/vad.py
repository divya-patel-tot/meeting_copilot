import numpy as np
import onnxruntime as ort


class SileroVAD:
    """Silero VAD ONNX wrapper (16 kHz, 512-sample chunks).

    Recurrent state persists across consecutive get_speech_probability() calls.
    Call reset_state() once at the start of each new listening session.
    """

    CHUNK_SIZE = 512
    SAMPLE_RATE = 16000
    STATE_SHAPE = (2, 1, 128)
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        self._input_name = inputs[0].name
        self._state_name = inputs[1].name
        self._sr_name = inputs[2].name

        self._sr = np.array([self.SAMPLE_RATE], dtype=np.int64)
        self.reset_state()

    def reset_state(self) -> None:
        """Re-initialize recurrent state for a new listening session."""
        self._state = np.zeros(self.STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)

    @staticmethod
    def prepare_chunk(
        audio_chunk: np.ndarray,
        *,
        normalize_min_peak: float = 0.05,
    ) -> np.ndarray:
        """Peak-normalize only when the signal is loud enough to be real speech."""
        chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
        if len(chunk) < SileroVAD.CHUNK_SIZE:
            chunk = np.pad(chunk, (0, SileroVAD.CHUNK_SIZE - len(chunk)))
        else:
            chunk = chunk[: SileroVAD.CHUNK_SIZE]

        peak = float(np.abs(chunk).max())
        if peak >= normalize_min_peak:
            chunk = (chunk / peak) * 0.95
        return chunk

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Run VAD on one 512-sample chunk; state carries over to the next call."""
        chunk = self.prepare_chunk(audio_chunk)

        # Streaming ONNX model: 64-sample context + 512-sample window.
        model_input = np.concatenate([self._context, chunk]).reshape(1, -1)
        self._context = chunk[-self.CONTEXT_SAMPLES :].copy()

        outputs = self.session.run(
            None,
            {
                self._input_name: model_input,
                self._state_name: self._state,
                self._sr_name: self._sr,
            },
        )
        self._state = np.copy(outputs[1])
        return float(outputs[0].reshape(-1)[0])
