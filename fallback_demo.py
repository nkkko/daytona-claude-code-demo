from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig

# Initialize the Daytona client with your API key and URL
config = DaytonaConfig(
    api_key="dtn_9a5695de4d3a86789d9a0cbc2f9ed8ff0b0ba2645e7d89e02093db92b52bfce9",
    server_url="https://app.daytona.io/api",
    target="us"
)
daytona = Daytona(config)

# Create the Sandbox instance
params = CreateWorkspaceParams(language="python")
print("Creating Daytona workspace...")
workspace = daytona.create(params)
print(f"Workspace created with ID: {workspace.id}")

# Python code to run inside the sandbox
# Using only standard library modules
python_code = """
import random
import math
import time
import statistics
from collections import Counter
import itertools

# Start timing
start_time = time.time()

# Generate large random dataset
print("Generating random data...")
data_size = 1000000
random_data = [random.random() * 100 for _ in range(data_size)]

# Basic statistics
print("Calculating statistics...")
data_mean = sum(random_data) / len(random_data)
data_min = min(random_data)
data_max = max(random_data)

# Calculate variance and standard deviation manually
variance = sum((x - data_mean) ** 2 for x in random_data) / len(random_data)
std_dev = math.sqrt(variance)

# Using statistics module
median = statistics.median(random_data[:10000])  # Using a sample for median calculation
mode_data = statistics.mode(random.choices(random_data, k=5000))

# Monte Carlo simulation for estimating π
print("Running Monte Carlo simulation to estimate π...")
inside_circle = 0
total_points = 1000000

for _ in range(total_points):
    x, y = random.random(), random.random()
    if (x**2 + y**2) <= 1:
        inside_circle += 1

pi_estimate = 4 * inside_circle / total_points

# Prime number calculation
print("Finding prime numbers...")
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Find primes up to 20000
primes = []
max_check = 20000
for num in range(2, max_check + 1):
    if is_prime(num):
        primes.append(num)

# Fibonacci sequence calculation
print("Generating Fibonacci sequence...")
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

fibonacci_numbers = fibonacci(50)

# Perform frequency analysis on random data
print("Performing frequency analysis...")
# Create bins for the random data
bins = {}
for i in range(10):
    lower = i * 10
    upper = (i + 1) * 10
    bins[f"{lower}-{upper}"] = 0

for num in random_data:
    bin_idx = min(9, int(num / 10))
    lower = bin_idx * 10
    upper = (bin_idx + 1) * 10
    bins[f"{lower}-{upper}"] += 1

# Simulate dice rolls
print("Simulating dice rolls...")
dice_rolls = [random.randint(1, 6) for _ in range(1000000)]
dice_frequencies = Counter(dice_rolls)

# Calculate probability of getting a specific dice sum
dice_sum_counts = Counter([random.randint(1, 6) + random.randint(1, 6) for _ in range(1000000)])
dice_sum_probabilities = {k: v/1000000 for k, v in dice_sum_counts.items()}

# Calculate combinations
print("Calculating combinations...")
def combination(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

combinations_results = [combination(20, k) for k in range(21)]

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Print results
print("\\nResults:")
print(f"Data size: {data_size} random numbers")
print(f"Mean: {data_mean:.6f}")
print(f"Standard Deviation: {std_dev:.6f}")
print(f"Min: {data_min:.6f}")
print(f"Max: {data_max:.6f}")
print(f"Median (10k sample): {median:.6f}")
print(f"Mode (5k sample): {mode_data:.6f}")
print(f"Pi estimate from {total_points} Monte Carlo samples: {pi_estimate:.10f}")
print(f"Error from actual π: {abs(pi_estimate - math.pi):.10f}")
print(f"Found {len(primes)} prime numbers below {max_check}")
print(f"Fibonacci sequence (first 10 numbers): {fibonacci_numbers[:10]}")
print(f"50th Fibonacci number: {fibonacci_numbers[-1]}")

print("\\nRandom data frequency distribution:")
for bin_range, count in bins.items():
    print(f"{bin_range}: {count} ({count/data_size*100:.2f}%)")

print("\\nDice roll probabilities:")
for sum_val, prob in sorted(dice_sum_probabilities.items()):
    print(f"Sum {sum_val}: {prob:.6f}")

print(f"\\nNumber of ways to choose 10 items from a set of 20: {combinations_results[10]}")
print(f"Total execution time: {execution_time:.2f} seconds")
"""

# Run the code securely inside the Sandbox
print("\nRunning calculations with standard library in the sandbox...")
response = workspace.process.code_run(python_code)
if response.exit_code != 0:
    print(f"Error running code: {response.exit_code} {response.result}")
else:
    print("\n--- SANDBOX OUTPUT ---")
    print(response.result)
    print("--- END OUTPUT ---")

# Clean up the Sandbox
print("\nCleaning up the sandbox...")
daytona.remove(workspace)
print("Sandbox removed successfully!")