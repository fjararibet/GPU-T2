import subprocess

# Define the command and its arguments


for n in range(100, 1000 + 1, 100):
    for m in range(100, 1000 + 1, 100):
        command = [
            "./build/GameOfLifeTest",
            "--workgroup-x", "32",
            "--workgroup-y", "32",
            "--seconds", "10",
            "--cuda",
            "--local",
            "-n", str(n),
            "-m", str(m),
        ]
        try:
            print(f"running for n: {n} m: {m}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            cells_per_sec = int(result.stdout.split()[0])
            with open("cuda_local_results.csv", "a") as f:
                f.write(f"{n},{m},{cells_per_sec}\n")
        except subprocess.CalledProcessError as e:
            print("Command failed with return code", e.returncode)
            print("Output:\n", e.stdout)
            print("Errors:\n", e.stderr)
