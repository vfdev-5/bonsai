"""Test from_pretrained method for VGG model."""
import jax.numpy as jnp
from absl.testing import absltest

from bonsai.models.vgg19 import modeling as model_lib


class TestFromPretrained(absltest.TestCase):
    """Test from_pretrained classmethod."""

    def test_from_pretrained_with_known_model(self):
        """Test loading a known model with from_pretrained."""
        model_name = "keras/vgg_19_imagenet"
        model = model_lib.VGG.from_pretrained(model_name)
        
        # Check that model is not None and is an instance of VGG
        self.assertIsNotNone(model)
        self.assertIsInstance(model, model_lib.VGG)
        
        # Test forward pass with dummy input
        batch_size, channels, image_size = 2, 3, 224
        dummy_input = jnp.ones((batch_size, image_size, image_size, channels), dtype=jnp.float32)
        logits = model(dummy_input)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, 1000))

    def test_from_pretrained_with_unknown_model(self):
        """Test that unknown model names raise ValueError."""
        with self.assertRaises(ValueError) as context:
            model_lib.VGG.from_pretrained("unknown/model-name")
        
        self.assertIn("unknown", str(context.exception).lower())

    def test_from_pretrained_with_custom_config(self):
        """Test loading with custom config."""
        model_name = "keras/vgg_19_imagenet"
        config = model_lib.ModelConfig.vgg_19()
        # Modify config to have custom num_classes
        config = model_lib.ModelConfig(num_classes=10, block_layers=config.block_layers, channels=config.channels)
        model = model_lib.VGG.from_pretrained(model_name, config=config)
        
        # Check that model uses custom config
        self.assertIsNotNone(model)
        
        # Test forward pass with custom output size
        batch_size, channels, image_size = 2, 3, 224
        dummy_input = jnp.ones((batch_size, image_size, image_size, channels), dtype=jnp.float32)
        logits = model(dummy_input)
        
        # Check output shape matches custom num_classes
        self.assertEqual(logits.shape, (batch_size, 10))


if __name__ == "__main__":
    absltest.main()
