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

# Check if we can install packages, but continue with standard library calculations if not
print("Attempting to install packages (will fall back to standard library if this fails)...")
install_code = """
import sys
import subprocess

try:
    # Try to get pip first
    subprocess.check_call(["curl", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"])
    subprocess.check_call([sys.executable, "get-pip.py", "--break-system-packages"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib", "--break-system-packages"])
    print("Packages successfully installed!")
    import numpy as np
    import matplotlib
    print(f"Using numpy version: {np.__version__}")
    print(f"Using matplotlib version: {matplotlib.__version__}")
    has_packages = True
except Exception as e:
    print(f"Failed to install packages: {e}")
    print("Will use standard library instead")
    has_packages = False

print(f"HAS_PACKAGES={has_packages}")
"""

install_response = workspace.process.code_run(install_code)
print(install_response.result)

# Check if packages were installed successfully
has_packages = "HAS_PACKAGES=True" in install_response.result

if has_packages:
    print("Successfully installed packages. Will use numpy and matplotlib.")
else:
    print("Could not install packages. Will use standard library.")

# Python code to run inside the sandbox
python_code = """
import time
import math
import random
import sys
from io import BytesIO
import base64

# Check if we have numpy and matplotlib
try:
    import numpy as np
    import matplotlib.pyplot as plt
    has_packages = True
    print("Using numpy and matplotlib for advanced calculations")
except ImportError:
    has_packages = False
    print("Using standard library for calculations")

# Start timing
start_time = time.time()

if has_packages:
    # Advanced calculations with numpy and matplotlib
    # Generate random data
    print("Generating random matrix data...")
    matrix_size = 500
    random_matrix_a = np.random.rand(matrix_size, matrix_size)
    random_matrix_b = np.random.rand(matrix_size, matrix_size)

    # Matrix operations
    print(f"Performing matrix multiplication on {matrix_size}x{matrix_size} matrices...")
    matrix_product = np.dot(random_matrix_a, random_matrix_b)

    # Calculate eigenvalues
    print("Calculating eigenvalues...")
    eigenvalues = np.linalg.eigvals(matrix_product)

    # Statistical analysis
    print("Performing statistical analysis...")
    mean_values = np.mean(matrix_product, axis=0)
    std_values = np.std(matrix_product, axis=0)
    max_values = np.max(matrix_product, axis=0)
    min_values = np.min(matrix_product, axis=0)

    # Monte Carlo simulation for estimating π
    print("Running Monte Carlo simulation to estimate π...")
    num_samples = 1000000
    points = np.random.rand(num_samples, 2)
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    points_inside_circle = np.sum(distances <= 1)
    pi_estimate = 4 * points_inside_circle / num_samples

    # Calculate prime numbers
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

    prime_count = 0
    max_prime = 50000
    for num in range(max_prime):
        if is_prime(num):
            prime_count += 1

    # Create visualization
    print("Creating visualizations...")
    plt.figure(figsize=(12, 10))

    # Plot 1: Eigenvalue distribution
    plt.subplot(2, 2, 1)
    plt.hist(np.real(eigenvalues), bins=50, alpha=0.7, color='blue')
    plt.title('Eigenvalue Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot 2: Matrix Heat Map
    plt.subplot(2, 2, 2)
    plt.imshow(matrix_product[:50, :50], cmap='viridis')
    plt.colorbar()
    plt.title('Matrix Product Heatmap (First 50x50)')

    # Plot 3: Mean values
    plt.subplot(2, 2, 3)
    plt.plot(mean_values[:100], color='green')
    plt.fill_between(range(100), mean_values[:100] - std_values[:100], 
                    mean_values[:100] + std_values[:100], alpha=0.3, color='green')
    plt.title('Mean Values with Standard Deviation')
    plt.xlabel('Column Index')
    plt.ylabel('Value')

    # Plot 4: Monte Carlo Pi Estimation
    plt.subplot(2, 2, 4)
    circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color='red')
    sample_points = np.random.rand(1000, 2)
    inside = np.sqrt((sample_points[:, 0] - 0.5)**2 + (sample_points[:, 1] - 0.5)**2) <= 0.5
    plt.scatter(sample_points[inside, 0], sample_points[inside, 1], s=1, color='blue')
    plt.scatter(sample_points[~inside, 0], sample_points[~inside, 1], s=1, color='red')
    plt.gca().add_patch(circle)
    plt.title(f'Monte Carlo π Estimation: {pi_estimate:.6f}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()

    # Save figure to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    
    # Print results
    print("\\nResults:")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Number of eigenvalues calculated: {len(eigenvalues)}")
    print(f"Mean of product matrix: {np.mean(matrix_product):.6f}")
    print(f"Pi estimate from {num_samples} Monte Carlo samples: {pi_estimate:.10f}")
    print(f"Error from actual π: {abs(pi_estimate - math.pi):.10f}")
    print(f"Found {prime_count} prime numbers below {max_prime}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("\\nImage data available as base64 string.")
    image_preview = image_base64[:100] + "..." + image_base64[-100:]
    print(f"Image preview: {image_preview}")

else:
    # Standard library calculations
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
    
    # End timing
    execution_time = time.time() - start_time
    
    # Print results
    print("\\nResults:")
    print(f"Data size: {data_size} random numbers")
    print(f"Mean: {data_mean:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Min: {data_min:.6f}")
    print(f"Max: {data_max:.6f}")
    print(f"Pi estimate from {total_points} Monte Carlo samples: {pi_estimate:.10f}")
    print(f"Error from actual π: {abs(pi_estimate - math.pi):.10f}")
    print(f"Found {len(primes)} prime numbers below {max_check}")
    print(f"Fibonacci sequence (first 10 numbers): {fibonacci_numbers[:10]}")
    print(f"50th Fibonacci number: {fibonacci_numbers[-1]}")
    print(f"Total execution time: {execution_time:.2f} seconds")
"""

# Run the code securely inside the Sandbox
print("\nRunning impressive calculations in the sandbox...")
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