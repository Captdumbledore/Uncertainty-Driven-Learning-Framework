from experience_providers.counterexample_provider import CounterexampleProvider

# Dummy dataset
dataset = [
    "image_0",
    "image_1",
    "image_2",
    "image_3",
    "image_4"
]
# Dummy confidence values
confidences = [
    0.95,
    0.72,
    0.98,
    0.61,
    0.89
]
# Create the provider
provider = CounterexampleProvider(dataset, confidences)
counterexamples = provider.get_counterexamples()
print("Counterexamples:", counterexamples)
print("Dataset:", provider.dataset)
print("Confidences:", provider.confidences)