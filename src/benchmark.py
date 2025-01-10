import subprocess
import csv
import hashlib

# Python skripts pwcracker programmas 'benchmarkosanai' (etalonuzdevuma izpildei)

# Palaisanas konfiguraciju parametri

files = ["1_mil_random_passwords.txt","10_k_random_passwords.txt","100_mil_random_passwords.txt"]
modes = ["--cpu", "--gpu"]
quartiles = ["1st", "2nd", "3rd", "4th"]
pw_status = ["found", "not_found"]
output = "benchmark_results.csv"

# Iegust konkretu paroli no dotas kvartiles tipa (1st, 2nd, 3rd, 4th)
def get_pw_by_quartile(file_path, quartile):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)
    
    index = {
        "1st": total_lines // 4,
        "2nd": total_lines // 2,
        "3rd": (3 * total_lines) // 4,
        "4th": total_lines - 1
    }[quartile]
    
    return lines[index].strip()

# SHA256 no dotas paroles, ka heksadecimals teksts
def get_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Izpildes laika merisana, pec konfiguracijas
def time_pwcracker(mode, file, password_hash):
    try:
        result = subprocess.run(
            ["/usr/bin/time", "-f", "%e", "./pwcracker", mode, file, password_hash],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=True
        )
        print(result.stdout)
        return result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(e.stdout) 
        return e.stderr.strip()

with open(output, mode='w', newline='') as csv_file:
    csv_wr = csv.writer(csv_file)
    csv_wr.writerow(["mode", "file", "password_status", "position", "execution_time"])

    # iziet cauri visam konfiguraciju permutacijam un ieraksta csv faila 
    for mode in modes:
        for file in files:
            for status in pw_status:
                if status == "found":
                    for position in quartiles:
                        password = get_pw_by_quartile(file, position)
                        password_hash = get_hash(password)
                        exec_time = time_pwcracker(mode, file, password_hash)
                        csv_wr.writerow([mode, file, status, position, exec_time])
                        print(f"Done: {mode}, {file}, {status}, {position}")
                else:
                    fake_hash = get_hash("nonexistantpassword")
                    exec_time = time_pwcracker(mode, file, fake_hash)
                    csv_wr.writerow([mode, file, status, "-", exec_time])
                    print(f"Done: {mode}, {file}, {status}")

print(f"Completed benchmark, output saved to {output}.")

