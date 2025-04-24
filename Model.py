from Components.model import TransformerWithFeatures, EncoderWithFeatures, Encoder, DecoderBlock, Decoder, TransformerBlock
import torch

class PyCode:


    def __init__(self):
        """
                        Initialize the model and load vocabulary
                        Moke sure you have installed the packages from requirements.txt

                        Returns:
                            None
                        """
        try:
            self.model = torch.load('./Components/Model.pt', weights_only=True)
        except:
            self.model = TransformerWithFeatures()
            state = torch.load('Components/model.pt', weights_only=True)
            self.model = self.model.load_state_dict(state)

    def generate(self, input_text, method, **kwargs):
        """
                Complete inference pipeline for code generation.

                Args:
                    input_text: The text prompt for code generation
                    method: Generation method ("greedy", "beam_search", or "sampling")
                    **kwargs: Additional parameters for the specific generation method
                        - For greedy: max_gen_len (default=512)
                        - For beam_search: beam_width (default=5), max_gen_len (default=512), length_penalty (default=1.0)
                        - For sampling: max_gen_len (default=512), temperature (default=0.8), top_p (default=0.9)

                Returns:
                    Generated code as a string
                """
        print(model.generate(input_text, method = method, **kwargs))

