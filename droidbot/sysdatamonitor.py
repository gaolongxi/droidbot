import csv
import os
import subprocess
import threading
import time
import re

class SysDataMonitor:
    def __init__(self, package_name, device_serial, output_dir, interval=10):
        self.package_name = package_name
        self.device_serial = device_serial
        self.output_dir = output_dir
        self.interval = interval
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.csv_filename_mem = os.path.join(output_dir, "data_mem.csv")
        self.csv_filename_cpu = os.path.join(output_dir, "data_cpu.csv")
        self.csv_filename_gpu = os.path.join(output_dir, "data_gpu.csv")

    def get_memory_usage(self):
        cmd = f'adb -s {self.device_serial} shell dumpsys meminfo {self.package_name}'
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(result.stdout)
        return result.stdout
    
    def parse_memory_usage(self, memory_usage):
        # print("Debug - Memory Usage Output:\n", memory_usage)
        def extract_section_data(section_name):
            pattern = rf"{section_name}\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
            # print(f"Debug - Regex Pattern for {section_name}: ", pattern)
            match = re.search(pattern, memory_usage)
            result = list(match.groups()) if match else [None] * 7
            # print(f"Debug - Result for {section_name}: ", result)
            return result
        
        native_heap_data = extract_section_data("Native Heap")
        dalvik_heap_data = extract_section_data("Dalvik Heap")
        app_summary_pattern = r"App Summary\s+Pss\(KB\)\s+[-]+\s+Java Heap:\s+(\d+)\s+Native Heap:\s+(\d+)\s+Code:\s+(\d+)\s+Stack:\s+(\d+)\s+Graphics:\s+(\d+)\s+Private Other:\s+(\d+)\s+System:\s+(\d+)"
        app_summary_match = re.search(app_summary_pattern, memory_usage)
        app_summary_data = app_summary_match.groups() if app_summary_match else [None] * 7

        # print(native_heap_data + dalvik_heap_data + list(app_summary_data))
        return native_heap_data + dalvik_heap_data + list(app_summary_data)
    
    def append_to_csv_mem(self, filename, data, headers):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({headers[i]: data[i] for i in range(len(data))})

    def get_cpu_usage(self):
        cmd = f'adb -s {self.device_serial} shell top -b -n 1 | grep {self.package_name}'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(result.stdout)
        return result.stdout
    
    def parse_cpu_usage(self, cpu_usage):
        lines = cpu_usage.strip().split('\n')
        parsed_data = []
        for line in lines:
            data = re.split(r'\s+', line)
            if len(data) >= 12 and data[11].startswith(self.package_name):
                parsed_data.append(data[:12])
        # print(parsed_data)
        return parsed_data
    
    def append_to_csv_cpu(self, filename, data_rows, headers):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            for data in data_rows:
                writer.writerow({headers[i]: data[i] for i in range(len(data))})

    def get_gpu_usage(self):
        cmd = f'adb -s {self.device_serial} shell dumpsys gfxinfo {self.package_name}'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(result.stdout)
        return result.stdout
    
    def parse_gpu_usage(self, gpu_usage):
        def find_metric(pattern):
            match = re.search(pattern, gpu_usage)
            return match.group(1) if match else None
        
        total_frames_rendered = find_metric(r"Total frames rendered: (\d+)")
        janky_frames = find_metric(r"Janky frames: (\d+)")
        janky_frames_percentage = find_metric(r"Janky frames: \d+ \((\d+\.\d+)%\)")

        percentile_50th = find_metric(r"50th percentile: (\d+)ms")
        percentile_90th = find_metric(r"90th percentile: (\d+)ms")
        percentile_95th = find_metric(r"95th percentile: (\d+)ms")
        percentile_99th = find_metric(r"99th percentile: (\d+)ms")

        total_gpu_memory_usage = find_metric(r"Total GPU memory usage: (\d+) bytes")

        # print([total_frames_rendered, janky_frames, janky_frames_percentage, 
                # percentile_50th, percentile_90th, percentile_95th, percentile_99th, 
                # total_gpu_memory_usage])
        return [total_frames_rendered, janky_frames, janky_frames_percentage, 
                percentile_50th, percentile_90th, percentile_95th, percentile_99th, 
                total_gpu_memory_usage]


    def append_to_csv_gpu(self, filename, data, headers):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({headers[i]: data[i] for i in range(len(data))})


    def start(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        while self.running:
            timestamp = time.time()
            
            # memory usage
            memory_usage = self.get_memory_usage()
            parsed_mem_data = self.parse_memory_usage(memory_usage)
            mem_headers = [
                'NativeHeap_Pss', 'NativeHeap_PrivateDirty', 'NativeHeap_PrivateClean', 
                'NativeHeap_SwapDirty', 'NativeHeap_HeapSize', 'NativeHeap_HeapAlloc', 
                'NativeHeap_HeapFree', 'DalvikHeap_Pss', 'DalvikHeap_PrivateDirty', 
                'DalvikHeap_PrivateClean', 'DalvikHeap_SwapDirty', 'DalvikHeap_HeapSize', 
                'DalvikHeap_HeapAlloc', 'DalvikHeap_HeapFree', 'JavaHeap', 'NativeHeap', 
                'Code', 'Stack', 'Graphics', 'PrivateOther', 'System'
            ]
            self.append_to_csv_mem(self.csv_filename_mem, [timestamp] + parsed_mem_data, ['Timestamp'] + mem_headers)

            # cpu usage
            cpu_usage = self.get_cpu_usage()
            parsed_cpu_data = self.parse_cpu_usage(cpu_usage)
            cpu_headers = ['Timestamp', 'PID', 'User', 'Priority', 'NI', 'VIRT', 'RES', 'SHR', 'S', 'CPU%', 'MEM%', 'Time', 'Command']
            parsed_cpu_data_with_timestamp = [[timestamp] + row for row in parsed_cpu_data]
            self.append_to_csv_cpu(self.csv_filename_cpu, parsed_cpu_data_with_timestamp, cpu_headers)  

            # gpu usage
            gpu_usage = self.get_gpu_usage()
            parsed_gpu_data = self.parse_gpu_usage(gpu_usage)
            gpu_headers = ['Timestamp', 'TotalFramesRendered', 'JankyFrames', 'JankyFramesPercentage', 
                            'Percentile50th', 'Percentile90th', 'Percentile95th', 'Percentile99th', 
                            'TotalGPUMemoryUsage']
            self.append_to_csv_gpu(self.csv_filename_gpu, [timestamp] + parsed_gpu_data, gpu_headers)

            time.sleep(self.interval)
