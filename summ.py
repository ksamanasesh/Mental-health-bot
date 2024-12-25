import time

# Start time
start_time = time.time()

# Run your code block with a continuous timer
for i in range(1000000):
    # Simulate some processing (your code logic)
    if i % 100000 == 0:  # Just for demonstration: simulate some progress
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Print the elapsed time, updating it in place every second
        print(f"\rElapsed time: {elapsed_time:.1f} seconds", end="")

        # Sleep for 1 second before updating again
        time.sleep(1)

# Final output after completion
elapsed_time = time.time() - start_time
print(f"\nTotal execution time: {elapsed_time:.1f} seconds")
