import sys
import importlib.util

print("Current sys.path:")
for p in sys.path:
    print("  ", p)

# Try to find the spec for the module imported as "cardiac_examples.electrophysio.data_processing"
spec = importlib.util.find_spec("cardiac_examples.electrophysio.data_processing")
print("\nSpec for 'cardiac_examples.electrophysio.data_processing':")
print("  ", spec)
if spec:
    print("Module origin:", spec.origin)

# Try to find the spec for the module imported as "electrophysio.data_processing"
spec2 = importlib.util.find_spec("electrophysio.data_processing")
print("\nSpec for 'electrophysio.data_processing':")
print("  ", spec2)
if spec2:
    print("Module origin:", spec2.origin)