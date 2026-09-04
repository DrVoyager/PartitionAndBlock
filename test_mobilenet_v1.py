import unittest

import torch

from mobilenet_v1 import IntermediateFeatureEncoder, PartitionAndBlockingModel, scale_channels


class ScaledProtectedEncoderTests(unittest.TestCase):
    def test_parameter_count_is_independent_of_block_size(self):
        counts = []
        for mask_side in (10, 16, 22, 28):
            model = PartitionAndBlockingModel(
                mask_side=mask_side,
                protected_width=1.0,
                protected_pool_size=2,
            )
            counts.append(sum(parameter.numel() for parameter in model.protected_layers.parameters()))
        self.assertEqual(counts, [380_864] * 4)

    def test_parameter_count_for_example_configurations(self):
        expected = {
            (0.5, 1): 81_504,
            (1.0, 2): 380_864,
            (2.0, 4): 2_027_136,
        }
        for (width, pool_size), parameter_count in expected.items():
            model = PartitionAndBlockingModel(
                mask_side=10,
                protected_width=width,
                protected_pool_size=pool_size,
            )
            self.assertEqual(sum(parameter.numel() for parameter in model.protected_layers.parameters()), parameter_count)

    def test_channel_scaling_rounds_to_multiples_of_eight(self):
        self.assertEqual(scale_channels(32, 0.1), 8)
        self.assertEqual(scale_channels(32, 0.5), 16)
        self.assertEqual(scale_channels(64, 1.0), 64)
        self.assertEqual(scale_channels(128, 2.0), 256)

    def test_ordinary_view_hides_only_the_protected_block(self):
        model = PartitionAndBlockingModel(mask_side=10, mask_fill="zero")
        smashed = torch.ones(1, 32, 32, 32)
        visible = model.mask_central_partition(smashed)
        self.assertEqual(torch.count_nonzero(visible[:, :, 11:21, 11:21]).item(), 0)
        self.assertEqual(torch.count_nonzero(visible).item(), 32 * (32 * 32 - 10 * 10))

    def test_learned_fill_is_input_independent(self):
        model = PartitionAndBlockingModel(mask_side=10, mask_fill="learned")
        with torch.no_grad():
            model.mask_fill.copy_(torch.arange(32).view(1, 32, 1, 1))
        first = model.mask_central_partition(torch.randn(1, 32, 32, 32))
        second = model.mask_central_partition(torch.randn(1, 32, 32, 32))
        expected = model.mask_fill.expand(1, 32, 10, 10)
        self.assertTrue(torch.equal(first[:, :, 11:21, 11:21], expected))
        self.assertTrue(torch.equal(second[:, :, 11:21, 11:21], expected))

    def test_feature_and_pool_shapes(self):
        encoder = IntermediateFeatureEncoder(
            input_channels=32,
            output_dim=256,
            r=1.0,
            pool_size=2,
        ).eval()
        with torch.no_grad():
            features = encoder.feature_extractor(torch.rand(2, 32, 10, 10))
            pooled = encoder.pool(features)
        self.assertEqual(features.shape, (2, 128, 3, 3))
        self.assertEqual(pooled.shape, (2, 128, 2, 2))

    def test_branch_and_model_output_shapes(self):
        model = PartitionAndBlockingModel(mask_side=28, protected_pool_size=2).eval()
        with torch.no_grad():
            smashed = model.client_model(torch.randn(2, 3, 32, 32))
            protected = model.extract_central_partition(smashed)
            protected_output = model.protected_layers(protected)
            logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(protected.shape, (2, 32, 28, 28))
        self.assertEqual(protected_output.shape, (2, 256))
        self.assertEqual(logits.shape, (2, 10))

    def test_invalid_pool_size_is_rejected(self):
        with self.assertRaises(ValueError):
            PartitionAndBlockingModel(mask_side=10, protected_pool_size=0)

    def test_invalid_width_is_rejected(self):
        with self.assertRaises(ValueError):
            PartitionAndBlockingModel(mask_side=10, protected_width=0)

    def test_pool_size_larger_than_feature_map_is_supported(self):
        model = PartitionAndBlockingModel(mask_side=10, protected_pool_size=4).eval()
        with torch.no_grad():
            protected = torch.randn(2, 32, 10, 10)
            features = model.protected_layers.feature_extractor(protected)
            pooled = model.protected_layers.pool(features)
            output = model.protected_layers(protected)
        self.assertEqual(features.shape, (2, 128, 3, 3))
        self.assertEqual(pooled.shape, (2, 128, 4, 4))
        self.assertEqual(output.shape, (2, 256))


if __name__ == "__main__":
    unittest.main()
