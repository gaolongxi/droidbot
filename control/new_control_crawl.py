import subprocess
import os
import threading
import queue
import logging
import traceback
import time

logging.basicConfig(filename='droidbot_crawl.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_event = threading.Event()

def connect_devices(start_ip, end_ip):
    devices = []
    logging.info("Starting to connect devices.")
    for i in range(start_ip, end_ip + 1):
        # ip = "192.168.1." + str(i)
        ip = f"10.129.47.131:{39060 + i}"
        try:
            logging.info(f"Attempting to connect to {ip}")
            process = subprocess.Popen(["adb", "connect", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            connect_result = (stdout + stderr).decode('utf-8')

            if "connected" in connect_result:
                logging.info(f"Connected to {ip}")
                process = subprocess.Popen(["adb", "devices"], stdout=subprocess.PIPE)
                stdout, _ = process.communicate()
                devices_info = stdout.decode('utf-8')

                for line in devices_info.splitlines():
                    if ip in line:
                        serial = line.split('\t')[0]
                        devices.append(serial)
                        logging.info(f"Device added: {serial}")
                        break
            else:
                logging.warning(f"Failed to connect to {ip}: {connect_result}")
        except Exception as e:
            logging.error(f"Exception during connection to {ip}: {e}")
            logging.error(traceback.format_exc())  # Log the full traceback

    logging.info(f"Finished connecting devices. Total devices connected: {len(devices)}")
    return devices

def run_droidbot_and_capture_output(cmd, output_file_path):
    with open(output_file_path, 'w') as file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                file.write(output.strip() + '\n')
                file.flush()

        process.wait()

unavailable_devices = set()

def is_device_online(device_id):
    process = subprocess.Popen(["adb", "-s", device_id, "get-state"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    return "device" in stdout.decode().strip()


def crawl_apk(device_id, apk_path, apk_queue, login_signal_dir, max_retries=3):
    apk_name = os.path.basename(apk_path).replace('.apk', '')
    output_dir = f"output/{device_id}/{apk_name}"
    log_file_path = f"{output_dir}/{apk_name}_log.txt"
    signal_file = os.path.join(login_signal_dir, f'login_complete_{device_id}.txt')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    if os.path.exists(signal_file):
        with open(signal_file, 'r') as file:
            if file.read().strip() == "login_required":
                logging.info(f"Waiting for manual login on {device_id} for {apk_name}")
                while True:
                    with open(signal_file, 'r') as file:
                        content = file.read().strip()
                        if content == 'done':
                            break
                    time.sleep(1)

    attempt = 0
    while attempt < max_retries:
        try:
            logging.info(f"Starting to crawl APK: {apk_name} on device: {device_id} (Attempt {attempt + 1})")
            cmd = [
                "droidbot",
                "-d", device_id,
                "-a", apk_path,
                "-o", output_dir,
                "-count", "1000"
            ]
            run_droidbot_and_capture_output(cmd, log_file_path)
            break
        except Exception as e:
            logging.error(f"An error occurred with device {device_id}: {e}")
            logging.error(traceback.format_exc())
            attempt += 1
            time.sleep(5)

    apk_queue.task_done()
    
    if stop_event.is_set():
        logging.info(f"Stopping crawling APK: {apk_name} on device: {device_id}")
        return

def device_worker(device_id, apk_queue, login_signal_dir):
    global unavailable_devices

    while not apk_queue.empty() and not stop_event.is_set():
        if device_id in unavailable_devices or not is_device_online(device_id)  :
            logging.info(f"Device {device_id} is unavailable. Skipping...")
            unavailable_devices.add(device_id)
            break

        try:
            apk_path = apk_queue.get_nowait()
            logging.info(f"Device {device_id} picked APK: {apk_path}")
            crawl_apk(device_id, apk_path, apk_queue, login_signal_dir, max_retries=3)
        except queue.Empty:
            logging.info(f"No more APKs to crawl for device {device_id}.")
            break
        except Exception as e:
            logging.error(f"An error occurred with device {device_id}: {e}")
            logging.error(traceback.format_exc())  # Log the full traceback
            break

logging.info("Starting the APK crawl process.")

def main():
    # devices = connect_devices(11, 70)
    devices = connect_devices(1, 2)
    login_signal_dir = "droidbot_signals"
    apk_queue = queue.Queue()
    apks_dir = "../APKPackage"
    apks = sorted(os.listdir(apks_dir))
    for apk in apks:
        if apk.endswith('.apk'):
            apk_queue.put(os.path.join(apks_dir, apk))

    threads = []
    for device in devices:
        thread = threading.Thread(target=device_worker, args=(device, apk_queue, login_signal_dir))
        # thread.daemon = True
        thread.start()
        threads.append(thread)

    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=1)  
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Stopping all threads.")
        stop_event.set()
    finally:
        logging.info("All threads have been signaled to stop. Waiting for clean termination...")
        for thread in threads:
            thread.join(timeout=5) 
        logging.info("All threads have been stopped.")

if __name__ == "__main__":
    main()

