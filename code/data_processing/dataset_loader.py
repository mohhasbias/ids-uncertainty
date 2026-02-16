#!/usr/bin/env python3
"""
Dataset loader for IDS uncertainty estimation experiments
Handles CIC-IDS2017, CIC-IDS2018, and mock datasets
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ids_models import GroundTruth


@dataclass
class IDSSample:
    """
    Single network flow sample for IDS classification
    """
    id: str
    description: str  # Human-readable description for LLM
    ground_truth: GroundTruth
    difficulty: str = "medium"
    features: Optional[Dict[str, Any]] = None  # Raw network features
    metadata: Optional[Dict[str, Any]] = None


class IDSDatasetLoader:
    """
    Loads IDS datasets in various formats
    Supports CIC-IDS JSONL format and mock datasets
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader with configuration

        Args:
            config: Dataset configuration from config.yaml
        """
        self.config = config
        self.file_path = Path(config["file"])
        self.format = config.get("format", "jsonl")
        self.fields = config.get("fields", {})
        self.defaults = config.get("defaults", {})

    def load(self, limit: Optional[int] = None) -> List[IDSSample]:
        """
        Load dataset samples

        Args:
            limit: Maximum number of samples to load (None = all)

        Returns:
            List of IDSSample objects
        """
        if self.format == "jsonl":
            return self._load_jsonl(limit)
        elif self.format == "json":
            return self._load_json(limit)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _load_jsonl(self, limit: Optional[int] = None) -> List[IDSSample]:
        """Load JSONL format dataset"""
        samples = []

        try:
            with open(self.file_path, 'r') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break

                    data = json.loads(line.strip())
                    sample = self._parse_sample(data)
                    if sample:
                        samples.append(sample)

        except FileNotFoundError:
            print(f"Warning: Dataset file not found: {self.file_path}")
            print("Please provide the dataset file or use mock data generation")
            return []

        return samples

    def _load_json(self, limit: Optional[int] = None) -> List[IDSSample]:
        """Load JSON array format dataset"""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)

            samples = []
            for i, item in enumerate(data):
                if limit and i >= limit:
                    break

                sample = self._parse_sample(item)
                if sample:
                    samples.append(sample)

            return samples

        except FileNotFoundError:
            print(f"Warning: Dataset file not found: {self.file_path}")
            print("Please provide the dataset file or use mock data generation")
            return []

    def _parse_sample(self, data: Dict[str, Any]) -> Optional[IDSSample]:
        """
        Parse raw data into IDSSample object

        Args:
            data: Raw data dictionary from dataset

        Returns:
            IDSSample object or None if parsing fails
        """
        try:
            # Extract fields using field mapping
            sample_id = self._get_field(data, "id", default=str(data.get("id", "unknown")))
            description = self._get_field(data, "description")
            label = self._get_field(data, "label")
            attack_type_str = self._get_field(data, "attack_type", default="unknown")
            features = self._get_field(data, "features", default=None)

            # Parse ground truth
            is_malicious = self._parse_label(label)
            attack_type = self._parse_attack_type(attack_type_str)
            threat_level = self._parse_threat_level(
                data.get("threat_level", self.defaults.get("threat_level"))
            )

            ground_truth = GroundTruth(
                is_malicious=is_malicious,
                attack_type=attack_type,
                threat_level=threat_level
            )

            # Get difficulty
            difficulty = data.get("difficulty", self.defaults.get("difficulty", "medium"))

            return IDSSample(
                id=sample_id,
                description=description,
                ground_truth=ground_truth,
                difficulty=difficulty,
                features=features,
                metadata=data.get("metadata")
            )

        except Exception as e:
            print(f"Warning: Failed to parse sample: {e}")
            return None

    def _get_field(self, data: Dict[str, Any], field_name: str, default: Any = None) -> Any:
        """
        Get field from data using field mapping

        Args:
            data: Raw data dictionary
            field_name: Logical field name (e.g., "id", "description")
            default: Default value if field not found

        Returns:
            Field value or default
        """
        # Get mapped field name from config
        mapped_name = self.fields.get(field_name)

        if mapped_name is None:
            # No mapping defined, try direct access
            return data.get(field_name, default)

        return data.get(mapped_name, default)

    def _parse_label(self, label: str) -> bool:
        """
        Parse label string to boolean (malicious or benign)

        Args:
            label: Label string from dataset

        Returns:
            True if malicious, False if benign
        """
        if not label:
            return False

        label_lower = label.lower().strip()
        benign_labels = ["benign", "normal", "legitimate", "0", "false"]

        return label_lower not in benign_labels

    def _parse_attack_type(self, attack_type_str: str) -> str:
        """
        Parse and normalize attack type string.

        Uses the dynamic enum system - returns normalized string that will be
        validated and auto-registered by the DynamicAttackType Pydantic validator.

        Args:
            attack_type_str: Attack type string from dataset

        Returns:
            Normalized attack type string
        """
        if not attack_type_str:
            return "unknown"

        # Normalize: lowercase, strip, replace spaces/hyphens with underscores
        attack_normalized = attack_type_str.lower().strip().replace(' ', '_').replace('-', '_')

        # Map common variations to canonical forms
        attack_mapping = {
            "normal": "benign",
            "normal_traffic": "benign",
            "portscan": "port_scan",
            "brute_force": "brute_force",
            "ssh_patator": "brute_force",
            "ftp_patator": "brute_force",
            "web_attack": "web_attack",
            "bot": "botnet",
            "bots": "botnet",
        }

        return attack_mapping.get(attack_normalized, attack_normalized)

    def _parse_threat_level(self, threat_str: Optional[str]) -> Optional[str]:
        """
        Parse and normalize threat level string.

        Uses the dynamic enum system - returns normalized string that will be
        validated and auto-registered by the DynamicThreatLevel Pydantic validator.

        Args:
            threat_str: Threat level string

        Returns:
            Normalized threat level string or None
        """
        if not threat_str:
            return None

        # Normalize: lowercase and strip
        return threat_str.lower().strip()


def generate_hard_mock_dataset(output_file: str = "datasets/mock_ids_hard.jsonl") -> List[IDSSample]:
    """
    Generate harder mock IDS dataset targeting 60-80% accuracy

    Distribution:
    - Easy: 20% (10 samples) - obvious attacks/benign
    - Medium: 40% (20 samples) - ambiguous, dual-interpretation
    - Hard: 40% (20 samples) - sophisticated or reverse traps

    Returns:
        List of 50 IDSSample objects with controlled difficulty
    """
    # Total: 50 samples
    hard_samples = [
        # ===== EASY (10 samples) - Obvious cases =====
        {
            "id": "1",
            "description": "SSH connection with 150 failed login attempts in 1 minute from IP 45.123.45.67",
            "label": "brute_force",
            "attack_type": "brute_force",
            "difficulty": "easy"
        },
        {
            "id": "2",
            "description": "Regular HTTPS traffic to corporate intranet portal during business hours",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "3",
            "description": "Massive flood of SYN packets to port 80, 50000 packets per second",
            "label": "dos",
            "attack_type": "dos",
            "difficulty": "easy"
        },
        {
            "id": "4",
            "description": "Employee accessing company email server with valid credentials during work hours",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "5",
            "description": "Sequential connection attempts to ports 1-65535 from single source IP",
            "label": "port_scan",
            "attack_type": "port_scan",
            "difficulty": "easy"
        },
        {
            "id": "6",
            "description": "Coordinated traffic spike from 5000+ distributed IP addresses targeting web server",
            "label": "ddos",
            "attack_type": "ddos",
            "difficulty": "easy"
        },
        {
            "id": "7",
            "description": "Automated database backup running scheduled task at 2 AM",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "8",
            "description": "HTTP POST with SQL UNION SELECT statement in user input field",
            "label": "web_attack",
            "attack_type": "web_attack",
            "difficulty": "easy"
        },
        {
            "id": "9",
            "description": "Network printer sending print job status updates to workstations",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "10",
            "description": "FTP server receiving 200 failed authentication attempts per minute",
            "label": "brute_force",
            "attack_type": "brute_force",
            "difficulty": "easy"
        },

        # ===== MEDIUM (20 samples) - Ambiguous, requires reasoning =====
        {
            "id": "11",
            "description": "High-volume HTTP requests to e-commerce site from single IP, 800 requests per minute during evening hours",
            "label": "benign",  # Could be legitimate crawler or bot
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "12",
            "description": "Database server processing 15000 queries per second from application tier",
            "label": "benign",  # High load but legitimate (Black Friday scenario)
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "13",
            "description": "SSH connection with 3 failed login attempts followed by successful authentication at 2 AM",
            "label": "benign",  # Admin fat-fingering password
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "14",
            "description": "Outbound traffic spike to multiple cloud storage providers, 500 MB transferred in 10 minutes",
            "label": "infiltration",  # Data exfiltration
            "attack_type": "infiltration",
            "difficulty": "medium"
        },
        {
            "id": "15",
            "description": "TCP connection to port 4444 with encrypted payload from internal workstation",
            "label": "infiltration",  # Port 4444 common for backdoors
            "attack_type": "infiltration",
            "difficulty": "medium"
        },
        {
            "id": "16",
            "description": "HTTP POST with 150 KB JSON payload to /api/search endpoint from mobile app",
            "label": "benign",  # Complex search query, not injection
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "17",
            "description": "Network scan detecting multiple open ports on newly deployed server in DMZ",
            "label": "benign",  # Vulnerability assessment by IT team
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "18",
            "description": "Repeated DNS queries for different subdomains of suspicious .tk domain",
            "label": "infiltration",  # DNS tunneling
            "attack_type": "infiltration",
            "difficulty": "medium"
        },
        {
            "id": "19",
            "description": "Web server receiving 1000 requests per second from data center IP range",
            "label": "dos",  # Volumetric attack from cloud VMs
            "attack_type": "dos",
            "difficulty": "medium"
        },
        {
            "id": "20",
            "description": "Internal host attempting connections to 30 different external IPs on port 443",
            "label": "benign",  # Browser with many tabs/extensions
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "21",
            "description": "Automated script making SSH connection attempts with 10 different key pairs",
            "label": "benign",  # Ansible/Terraform deployment automation
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "22",
            "description": "ICMP echo requests sent to entire /24 subnet from network monitoring tool",
            "label": "benign",  # Network discovery by monitoring system
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "23",
            "description": "Large number of HTTP redirects (302) looping between two internal web servers",
            "label": "benign",  # Misconfiguration, not attack
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "24",
            "description": "Workstation generating unusual TLS handshake patterns with varied cipher suites",
            "label": "infiltration",  # Malware testing defenses
            "attack_type": "infiltration",
            "difficulty": "medium"
        },
        {
            "id": "25",
            "description": "Multiple failed VPN authentication attempts from traveling employee's hotel WiFi",
            "label": "benign",  # Legitimate user with connectivity issues
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "26",
            "description": "HTTP traffic with unusual User-Agent string 'python-requests/2.28.0' to API endpoint",
            "label": "benign",  # Legitimate API client
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "27",
            "description": "File server experiencing rapid sequential file access across 1000+ files",
            "label": "infiltration",  # Ransomware encryption activity
            "attack_type": "infiltration",
            "difficulty": "medium"
        },
        {
            "id": "28",
            "description": "Burst of ARP requests from DHCP server after network segment restart",
            "label": "benign",  # Normal network recovery
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "29",
            "description": "Spike in DNS queries for NTP pool servers from multiple internal hosts",
            "label": "benign",  # Time synchronization
            "attack_type": "benign",
            "difficulty": "medium"
        },
        {
            "id": "30",
            "description": "Web application making 500 database connections per minute with short duration",
            "label": "web_attack",  # Connection pool exhaustion attack
            "attack_type": "web_attack",
            "difficulty": "medium"
        },

        # ===== HARD (20 samples) - Sophisticated or reverse traps =====
        {
            "id": "31",
            "description": "Periodic HTTPS connections to CDN every 30 minutes, consistent 2 KB payload size",
            "label": "infiltration",  # C2 beacon disguised as CDN traffic
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "32",
            "description": "Security scanner generating 2000 connection attempts across internal network",
            "label": "benign",  # Authorized penetration test
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "33",
            "description": "Internal workstation downloading PowerShell scripts from GitHub repository during work hours",
            "label": "infiltration",  # Malware dropper disguised as dev activity
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "34",
            "description": "Automated vulnerability scanner probing web application with SQL injection payloads",
            "label": "benign",  # Security testing, not real attack
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "35",
            "description": "DNS queries for long subdomain names with base64-like character patterns",
            "label": "infiltration",  # DNS exfiltration
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "36",
            "description": "Software update service connecting to 100+ IP addresses in rapid succession",
            "label": "benign",  # Legitimate CDN distribution network
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "37",
            "description": "Low-volume traffic to pastebin-like service with encrypted data in HTTP POST body",
            "label": "infiltration",  # Data staging before exfiltration
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "38",
            "description": "Continuous port scanning from security appliance to detect rogue devices",
            "label": "benign",  # Network access control (NAC) system
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "39",
            "description": "Spike in outbound email connections during business hours, 500 emails per hour",
            "label": "benign",  # Marketing campaign or newsletter
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "40",
            "description": "Unusual HTTP headers with encoded data in Cookie field making API requests",
            "label": "web_attack",  # Header injection attack
            "attack_type": "web_attack",
            "difficulty": "hard"
        },
        {
            "id": "41",
            "description": "Internal server making TLS connections to IP addresses without DNS resolution",
            "label": "infiltration",  # C2 communication bypassing DNS monitoring
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "42",
            "description": "Nmap scan from IT department testing firewall rules after configuration change",
            "label": "benign",  # Authorized network testing
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "43",
            "description": "HTTPS traffic with valid certificate to legitimate domain but unusual request timing patterns",
            "label": "infiltration",  # Compromised legitimate service used for C2
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "44",
            "description": "Kubernetes cluster generating thousands of internal API calls during scaling event",
            "label": "benign",  # Container orchestration, not attack
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "45",
            "description": "Workstation beacon traffic to cloud service every 15 minutes with small payloads",
            "label": "benign",  # Legitimate cloud app sync (Dropbox, OneDrive)
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "46",
            "description": "SSH traffic with multiple authentication methods attempted in sequence (key, password, 2FA)",
            "label": "benign",  # Misconfigured SSH client trying fallback methods
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "47",
            "description": "Web scraper making polite requests with 5-second delays across 10000 pages",
            "label": "benign",  # Legitimate search engine crawler
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "48",
            "description": "Periodic HTTP requests to social media APIs with OAuth tokens from internal application",
            "label": "infiltration",  # Stolen credentials being used for data harvesting
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "49",
            "description": "Burst of TCP connections to various ports on localhost from development environment",
            "label": "benign",  # Developer running integration tests
            "attack_type": "benign",
            "difficulty": "hard"
        },
        {
            "id": "50",
            "description": "Network monitoring tool generating SNMP queries to all devices every 5 minutes",
            "label": "benign",  # Network management, not reconnaissance
            "attack_type": "benign",
            "difficulty": "hard"
        }
    ]

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in hard_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated 50 hard mock samples and saved to {output_file}")
    print("\nDifficulty distribution:")
    print(f"  Easy: 10 samples (20%)")
    print(f"  Medium: 20 samples (40%)")
    print(f"  Hard: 20 samples (40%)")
    print("\nExpected LLM accuracy: 60-80% (30-40% of hard/medium samples will be misclassified)")

    # Parse and return IDSSample objects
    config = {
        "file": output_file,
        "format": "jsonl",
        "fields": {
            "id": "id",
            "description": "description",
            "label": "label",
            "attack_type": "attack_type"
        },
        "defaults": {
            "difficulty": "medium",
            "threat_level": "medium"
        }
    }

    loader = IDSDatasetLoader(config)
    return loader.load()


def generate_mock_dataset(n_samples: int = 10, output_file: str = "datasets/mock_ids_sample.jsonl") -> List[IDSSample]:
    """
    Generate mock IDS dataset for testing without real data

    Args:
        n_samples: Number of samples to generate
        output_file: Path to save generated dataset

    Returns:
        List of generated IDSSample objects
    """
    mock_samples = [
        {
            "id": "1",
            "description": "Normal HTTP GET request to corporate website during business hours",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "2",
            "description": "SSH connection with 50 failed login attempts in 2 minutes from unknown IP",
            "label": "brute_force",
            "attack_type": "brute_force",
            "difficulty": "easy"
        },
        {
            "id": "3",
            "description": "Large volume of SYN packets sent to web server without completing handshake",
            "label": "dos",
            "attack_type": "dos",
            "difficulty": "medium"
        },
        {
            "id": "4",
            "description": "Port scan detected: sequential connection attempts to ports 1-1024 from single source",
            "label": "port_scan",
            "attack_type": "port_scan",
            "difficulty": "medium"
        },
        {
            "id": "5",
            "description": "Regular database query from application server to backend database",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "6",
            "description": "SQL injection attempt detected in HTTP POST parameters with UNION SELECT",
            "label": "web_attack",
            "attack_type": "web_attack",
            "difficulty": "hard"
        },
        {
            "id": "7",
            "description": "Unusual outbound traffic at 3 AM to unknown external server with encrypted payload",
            "label": "infiltration",
            "attack_type": "infiltration",
            "difficulty": "hard"
        },
        {
            "id": "8",
            "description": "Email server sending routine authentication confirmations to users",
            "label": "benign",
            "attack_type": "benign",
            "difficulty": "easy"
        },
        {
            "id": "9",
            "description": "Coordinated DDoS attack from 1000+ IP addresses overwhelming DNS server",
            "label": "ddos",
            "attack_type": "ddos",
            "difficulty": "medium"
        },
        {
            "id": "10",
            "description": "Workstation making periodic IRC connections with command-and-control signatures",
            "label": "botnet",
            "attack_type": "botnet",
            "difficulty": "hard"
        }
    ]

    # Repeat samples to reach n_samples
    samples = []
    for i in range(n_samples):
        sample_data = mock_samples[i % len(mock_samples)].copy()
        sample_data["id"] = str(i + 1)
        samples.append(sample_data)

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {n_samples} mock samples and saved to {output_file}")

    # Parse and return IDSSample objects
    config = {
        "file": output_file,
        "format": "jsonl",
        "fields": {
            "id": "id",
            "description": "description",
            "label": "label",
            "attack_type": "attack_type"
        },
        "defaults": {
            "difficulty": "medium",
            "threat_level": "medium"
        }
    }

    loader = IDSDatasetLoader(config)
    return loader.load()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate mock IDS datasets")
    parser.add_argument(
        "--generate-hard",
        action="store_true",
        help="Generate harder mock dataset (50 samples, 60-80%% expected accuracy)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to generate (for easy dataset only)"
    )

    args = parser.parse_args()

    if args.generate_hard:
        print("Generating HARD mock IDS dataset (targeting 60-80%% accuracy)...")
        samples = generate_hard_mock_dataset()
    else:
        print("Generating easy mock IDS dataset...")
        samples = generate_mock_dataset(n_samples=args.samples)

    print(f"\nLoaded {len(samples)} samples")
    print("\nFirst 3 samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  ID: {sample.id}")
        print(f"  Description: {sample.description}")
        print(f"  Ground Truth: {sample.ground_truth.attack_type.value} (malicious={sample.ground_truth.is_malicious})")
        print(f"  Difficulty: {sample.difficulty}")
