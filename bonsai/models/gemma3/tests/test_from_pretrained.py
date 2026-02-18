"""Test from_pretrained method for Gemma3 model."""
import os

import jax.numpy as jnp
from absl.testing import absltest

from bonsai.models.gemma3 import modeling as model_lib


class TestFromPretrained(absltest.TestCase):
    """Test from_pretrained classmethod."""

    def test_from_pretrained_with_unknown_model(self):
        """Test that unknown model names raise ValueError."""
        with self.assertRaises(ValueError) as context:
            model_lib.Gemma3Model.from_pretrained("unknown/model-name")
        
        self.assertIn("unknown", str(context.exception).lower())

    def test_from_pretrained_requires_token(self):
        """Test that from_pretrained can be called with access_token parameter."""
        # This test just validates the method signature accepts the token parameter
        # We don't actually test loading because it requires a valid HF token
        
        # Just check that the method exists and has the right signature
        self.assertTrue(hasattr(model_lib.Gemma3Model, 'from_pretrained'))
        self.assertTrue(callable(model_lib.Gemma3Model.from_pretrained))
        
        # Check method signature includes expected parameters
        import inspect
        sig = inspect.signature(model_lib.Gemma3Model.from_pretrained)
        param_names = list(sig.parameters.keys())
        self.assertIn('model_name', param_names)
        self.assertIn('config', param_names)
        self.assertIn('norm_dtype', param_names)
        self.assertIn('access_token', param_names)


if __name__ == "__main__":
    absltest.main()
