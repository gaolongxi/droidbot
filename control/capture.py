import subprocess
import threading
import queue
import logging
import os

# Function to handle reading output from a subprocess
def read_output(pipe, queue):
    try:
        with pipe:
            for line in iter(pipe.readline, b''):
                queue.put(line.decode())
    finally:
        queue.put(None)

# Function to run droidbot in a subprocess and capture its output
def run_droidbot_and_capture_output(cmd, output_file_path):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    q = queue.Queue()
    t = threading.Thread(target=read_output, args=(process.stdout, q))
    t.daemon = True
    t.start()

    # Open the output file
    with open(output_file_path, 'w') as file:
        # Read output lines from the queue
        for line in iter(q.get, None):
            print("Droidbot Output:", line.strip())  # Print to console
            file.write(line)  # Write to file

    # Wait for the subprocess to finish
    process.wait()

# Example command to run droidbot
droidbot_cmd = ["droidbot", "-a", "../APKPackage/weibo.apk", "-o", "output_test_capture", "-count", "100"]
output_file = "droidbot_output.log"

# Running droidbot and capturing its output
run_droidbot_and_capture_output(droidbot_cmd, output_file)