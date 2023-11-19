import sys
import os

def signal_login_completion(ip_address):
    signal_file = os.path.join('droidbot_signals', f'login_complete_{ip_address}.txt')
    with open(signal_file, 'w') as file:
        file.write("done")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python signal_login_completion.py [ip_address]")
        sys.exit(1)

    ip_address = sys.argv[1]
    signal_login_completion(ip_address)
    print(f"Signaled login completion for device {ip_address}")
