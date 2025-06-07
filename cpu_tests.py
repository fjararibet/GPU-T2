import subprocess

# Define the command and its arguments
command = [
    "./build/GameOfLifeTest",
    "--workgroup-x", "16",
    "--workgroup-y", "16",
    "--seconds", "60",
    "--opencl"
]

# Run the command
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    cells_per_sec = int(result.stdout.split())
except subprocess.CalledProcessError as e:
    print("Command failed with return code", e.returncode)
    print("Output:\n", e.stdout)
    print("Errors:\n", e.stderr)
