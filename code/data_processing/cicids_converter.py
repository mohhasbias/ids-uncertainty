#!/usr/bin/env python3
"""
CIC-IDS2017 Feature-to-Text Converter

Converts numerical network flow features to natural language descriptions
for LLM-based intrusion detection experiments.

Strategy: Three-tier feature extraction
- Tier 1: Critical identifiers (always included)
- Tier 2: Attack-specific discriminative features
- Tier 3: Statistical summaries
"""

from typing import Dict, Any, List
import pandas as pd


class CICIDSConverter:
    """
    Converts CIC-IDS2017 and CIC-IDS2018 numerical features to natural language descriptions.

    Supports both dataset column naming conventions through flexible key lookup.
    """

    # CIC-IDS2018 → CIC-IDS2017 column name mapping
    # Allows converter to work with both dataset versions
    COLUMN_MAPPING = {
        # Port and basic flow info
        'Dst Port': 'Destination Port',

        # Forward packet metrics
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',

        # Backward packet metrics
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',

        # Flow rate metrics
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',

        # Inter-arrival time
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Bwd IAT Tot': 'Bwd IAT Total',

        # Header lengths
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',

        # Packet rates
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',

        # Packet length metrics
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',

        # Flag counts
        'FIN Flag Cnt': 'FIN Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',

        # Size metrics
        'Pkt Size Avg': 'Average Packet Size',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',

        # Window bytes
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',

        # Forward packet data
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
    }

    # CIC-IDS2018 Numeric Label → Attack Type Name Mapping
    # Inferred from distribution analysis and flow characteristics (Task 21)
    # Source: .taskmaster/docs/cicids2018-dataset-conversion.md
    CICIDS2018_LABEL_MAPPING = {
        1: "Benign",                    # 77.50% - Normal traffic
        2: "FTP-BruteForce",            #  3.09% - FTP brute-force attacks
        3: "SSH-Bruteforce",            #  2.98% - SSH brute-force attacks
        4: "DoS-Hulk",                  # 10.95% - DoS Hulk attacks
        5: "DoS-SlowHTTPTest",          #  4.59% - Slow HTTP test attacks
        6: "DoS-Slowloris",             #  0.67% - Slowloris attacks
        7: "DoS-GoldenEye",             #  0.18% - GoldenEye attacks
        8: "DDoS-LOIC-UDP",             #  0.03% - DDoS LOIC UDP attacks
        9: "Infiltration",              #  0.01% - Infiltration attacks
        10: "Bot",                      #  0.00% - Botnet traffic
        11: "WebAttack",                #  0.00% - Web attacks (SQL Injection/XSS)
    }

    # CIC-IDS2018 Attack Type → Standard Category Mapping
    CICIDS2018_TO_STANDARD = {
        "Benign": "benign",
        "FTP-BruteForce": "brute_force",
        "SSH-Bruteforce": "brute_force",
        "DoS-Hulk": "dos",
        "DoS-SlowHTTPTest": "dos",
        "DoS-Slowloris": "dos",
        "DoS-GoldenEye": "dos",
        "DDoS-LOIC-UDP": "ddos",
        "Infiltration": "infiltration",
        "Bot": "botnet",
        "WebAttack": "web_attack",
    }

    # Tier 1: Critical features (always included)
    CRITICAL_FEATURES = [
        'Destination Port',
        'Flow Duration',
        'Attack Type'
    ]

    # Tier 2: Attack-specific discriminative features
    ATTACK_FEATURES = {
        'DoS': [
            'Flow Bytes/s',
            'Flow Packets/s',
            'Total Fwd Packets',
            'Flow Duration'
        ],
        'DDoS': [
            'Flow Bytes/s',
            'Flow Packets/s',
            'Total Fwd Packets',
            'Destination Port'
        ],
        'PortScan': [
            'Destination Port',
            'Flow Duration',
            'ACK Flag Count',
            'Total Fwd Packets',
            'Fwd Packets/s'
        ],
        'Brute Force': [
            'Flow Duration',
            'Fwd Packets/s',
            'FIN Flag Count',
            'PSH Flag Count',
            'Total Fwd Packets'
        ],
        'Web Attack': [
            'Total Length of Fwd Packets',
            'Average Packet Size',
            'Destination Port',
            'Fwd Packet Length Mean'
        ],
        'Bot': [
            'Flow IAT Mean',
            'Active Mean',
            'Idle Mean',
            'Flow Duration',
            'Average Packet Size'
        ],
        'Normal Traffic': [
            'Flow Duration',
            'Average Packet Size',
            'Flow Packets/s',
            'Destination Port'
        ]
    }

    # Common port mappings for context
    WELL_KNOWN_PORTS = {
        20: 'FTP data',
        21: 'FTP control',
        22: 'SSH',
        23: 'Telnet',
        25: 'SMTP',
        53: 'DNS',
        80: 'HTTP',
        110: 'POP3',
        143: 'IMAP',
        443: 'HTTPS',
        445: 'SMB',
        3306: 'MySQL',
        3389: 'RDP',
        5432: 'PostgreSQL',
        8080: 'HTTP-alt'
    }

    def __init__(self):
        """Initialize the converter"""
        pass

    def _get_feature_value(self, flow: Dict[str, Any], key: str, default: Any = 0) -> Any:
        """
        Get feature value with fallback for column name variations.

        Supports both CIC-IDS2017 and CIC-IDS2018 naming conventions by:
        1. Trying the canonical CIC-IDS2017 key first
        2. Trying the mapped CIC-IDS2018 key if not found
        3. Returning default value if neither exists

        Args:
            flow: Dictionary of network flow features
            key: Canonical feature name (CIC-IDS2017 convention)
            default: Default value if feature not found (default: 0)

        Returns:
            Feature value or default

        Example:
            >>> flow = {"Dst Port": 80}  # CIC-IDS2018 format
            >>> converter._get_feature_value(flow, "Destination Port")
            80  # Successfully mapped from "Dst Port"
        """
        # Try direct key first (CIC-IDS2017 canonical name)
        if key in flow:
            return flow[key]

        # Try reverse lookup: find CIC-IDS2018 equivalent for this canonical name
        for ids2018_key, canonical_key in self.COLUMN_MAPPING.items():
            if canonical_key == key and ids2018_key in flow:
                return flow[ids2018_key]

        # Fallback to default
        return default

    def convert_flow_to_text(self, flow: Dict[str, Any]) -> str:
        """
        Convert a single network flow to natural language description

        Args:
            flow: Dictionary of network flow features (from CIC-IDS2017 or CIC-IDS2018)

        Returns:
            Natural language description string
        """
        attack_type = self._get_feature_value(flow, 'Attack Type', 'Normal Traffic')

        # Build description in parts
        parts = []

        # Part 1: Port and protocol context
        parts.append(self._describe_port_context(flow))

        # Part 2: Attack-specific characteristics
        parts.append(self._describe_attack_characteristics(flow, attack_type))

        # Part 3: Traffic statistics
        parts.append(self._describe_traffic_statistics(flow))

        # Part 4: TCP flags (if relevant)
        flag_desc = self._describe_tcp_flags(flow)
        if flag_desc:
            parts.append(flag_desc)

        # Combine parts into coherent description
        description = ' '.join(parts)

        return description

    def _describe_port_context(self, flow: Dict[str, Any]) -> str:
        """Describe destination port with service context"""
        port = int(self._get_feature_value(flow, 'Destination Port', 0))

        if port in self.WELL_KNOWN_PORTS:
            service = self.WELL_KNOWN_PORTS[port]
            return f"Network flow to destination port {port} ({service}):"
        elif port < 1024:
            return f"Network flow to privileged port {port}:"
        elif port > 49152:
            return f"Network flow to ephemeral port {port}:"
        else:
            return f"Network flow to destination port {port}:"

    def _describe_attack_characteristics(self, flow: Dict[str, Any], attack_type: str) -> str:
        """
        Describe attack-specific characteristics based on attack type

        Args:
            flow: Network flow features
            attack_type: Type of attack (or 'Normal Traffic')

        Returns:
            Description of attack-specific characteristics
        """
        if attack_type == 'DoS':
            return self._describe_dos(flow)
        elif attack_type == 'DDoS':
            return self._describe_ddos(flow)
        elif attack_type.startswith('PortScan'):
            return self._describe_portscan(flow)
        elif 'Brute Force' in attack_type or 'Patator' in attack_type:
            return self._describe_bruteforce(flow)
        elif 'Web Attack' in attack_type:
            return self._describe_webattack(flow)
        elif 'Bot' in attack_type:
            return self._describe_bot(flow)
        else:  # Normal Traffic
            return self._describe_normal(flow)

    def _describe_dos(self, flow: Dict[str, Any]) -> str:
        """Describe DoS attack characteristics"""
        packets_per_sec = self._get_feature_value(flow, 'Flow Packets/s', 0)
        bytes_per_sec = self._get_feature_value(flow, 'Flow Bytes/s', 0)
        total_packets = self._get_feature_value(flow, 'Total Fwd Packets', 0)

        # Convert to human-readable units
        mbps = bytes_per_sec * 8 / 1_000_000  # Convert to Mbps

        if packets_per_sec > 10000:
            intensity = "extremely high"
        elif packets_per_sec > 1000:
            intensity = "very high"
        else:
            intensity = "elevated"

        return f"{intensity} packet rate of {packets_per_sec:,.0f} packets/s with {mbps:.1f} Mbps throughput, total {total_packets:,.0f} packets"

    def _describe_ddos(self, flow: Dict[str, Any]) -> str:
        """Describe DDoS attack characteristics"""
        packets_per_sec = self._get_feature_value(flow, 'Flow Packets/s', 0)
        bytes_per_sec = self._get_feature_value(flow, 'Flow Bytes/s', 0)

        mbps = bytes_per_sec * 8 / 1_000_000

        return f"distributed attack pattern with {packets_per_sec:,.0f} packets/s ({mbps:.1f} Mbps), characteristic of coordinated traffic flooding"

    def _describe_portscan(self, flow: Dict[str, Any]) -> str:
        """Describe port scanning characteristics"""
        duration = self._get_feature_value(flow, 'Flow Duration', 0) / 1_000_000  # Convert to seconds
        fwd_packets = self._get_feature_value(flow, 'Total Fwd Packets', 0)
        ack_flags = self._get_feature_value(flow, 'ACK Flag Count', 0)

        if duration < 1:
            speed = "rapid"
        elif duration < 10:
            speed = "moderate"
        else:
            speed = "slow"

        return f"{speed} reconnaissance activity with {fwd_packets:,.0f} probe packets over {duration:.1f} seconds, ACK flags: {ack_flags}"

    def _describe_bruteforce(self, flow: Dict[str, Any]) -> str:
        """Describe brute force attack characteristics"""
        duration = self._get_feature_value(flow, 'Flow Duration', 0) / 1_000_000
        packets_per_sec = self._get_feature_value(flow, 'Fwd Packets/s', 0)
        total_packets = self._get_feature_value(flow, 'Total Fwd Packets', 0)

        return f"repeated authentication attempts with {packets_per_sec:.0f} packets/s over {duration:.1f} seconds, total {total_packets:,.0f} connection attempts"

    def _describe_webattack(self, flow: Dict[str, Any]) -> str:
        """Describe web attack characteristics"""
        fwd_bytes = self._get_feature_value(flow, 'Total Length of Fwd Packets', 0)
        avg_size = self._get_feature_value(flow, 'Average Packet Size', 0)
        fwd_mean = self._get_feature_value(flow, 'Fwd Packet Length Mean', 0)

        kb_transferred = fwd_bytes / 1024

        if fwd_mean > 1000:
            payload_desc = "large"
        elif fwd_mean > 500:
            payload_desc = "medium-sized"
        else:
            payload_desc = "small"

        return f"HTTP traffic with {payload_desc} payloads ({fwd_mean:.0f} bytes avg), {kb_transferred:.1f} KB total data transferred, average packet size {avg_size:.0f} bytes"

    def _describe_bot(self, flow: Dict[str, Any]) -> str:
        """Describe botnet traffic characteristics"""
        iat_mean = self._get_feature_value(flow, 'Flow IAT Mean', 0) / 1000  # Convert to ms
        active_mean = self._get_feature_value(flow, 'Active Mean', 0) / 1_000_000  # Convert to seconds
        idle_mean = self._get_feature_value(flow, 'Idle Mean', 0) / 1_000_000

        if iat_mean < 100:
            pattern = "rapid periodic"
        elif iat_mean < 1000:
            pattern = "steady periodic"
        else:
            pattern = "slow periodic"

        return f"{pattern} communication pattern with {iat_mean:.1f}ms inter-arrival time, active periods {active_mean:.1f}s, idle periods {idle_mean:.1f}s (potential C2 beacon)"

    def _describe_normal(self, flow: Dict[str, Any]) -> str:
        """Describe normal traffic characteristics"""
        duration = self._get_feature_value(flow, 'Flow Duration', 0) / 1_000_000
        avg_size = self._get_feature_value(flow, 'Average Packet Size', 0)
        packets_per_sec = self._get_feature_value(flow, 'Flow Packets/s', 0)

        if packets_per_sec < 10:
            activity = "low-volume"
        elif packets_per_sec < 100:
            activity = "moderate"
        else:
            activity = "high-volume"

        return f"{activity} legitimate traffic with {packets_per_sec:.0f} packets/s, average packet size {avg_size:.0f} bytes, flow duration {duration:.1f} seconds"

    def _describe_traffic_statistics(self, flow: Dict[str, Any]) -> str:
        """Describe general traffic statistics"""
        duration = self._get_feature_value(flow, 'Flow Duration', 0) / 1_000_000
        iat_mean = self._get_feature_value(flow, 'Flow IAT Mean', 0) / 1000  # to ms

        if duration < 1:
            timing = "Very short flow"
        elif duration < 10:
            timing = "Short flow"
        elif duration < 60:
            timing = "Medium duration flow"
        else:
            timing = "Long-lived flow"

        return f"{timing} ({duration:.2f}s) with {iat_mean:.2f}ms average inter-arrival time."

    def _describe_tcp_flags(self, flow: Dict[str, Any]) -> str:
        """Describe TCP flag patterns"""
        fin = int(self._get_feature_value(flow, 'FIN Flag Count', 0))
        psh = int(self._get_feature_value(flow, 'PSH Flag Count', 0))
        ack = int(self._get_feature_value(flow, 'ACK Flag Count', 0))

        # Only include if flags are present
        flags = []
        if fin > 0:
            flags.append(f"FIN={fin}")
        if psh > 0:
            flags.append(f"PSH={psh}")
        if ack > 0:
            flags.append(f"ACK={ack}")

        if flags:
            return f"TCP flags: {', '.join(flags)}."
        else:
            return ""

    def normalize_attack_type(self, attack_type) -> str:
        """
        Normalize CIC-IDS attack type labels to consistent format.

        Supports both CIC-IDS2017 (text labels) and CIC-IDS2018 (numeric labels).

        Two-step process for CIC-IDS2018:
        1. Convert numeric label (1-11) to attack type name
        2. Normalize attack type name to standard category

        Mappings:
        - CIC-IDS2017: "Normal Traffic" -> "benign"
        - CIC-IDS2018: 1 -> "Benign" -> "benign"

        Standard categories: benign, dos, ddos, port_scan, brute_force,
                           web_attack, botnet, infiltration

        Args:
            attack_type: Original attack type (str or int/float)
                        - String: CIC-IDS2017 format ("Normal Traffic", "DoS", etc.)
                        - Numeric: CIC-IDS2018 format (1, 2, 3, ..., 11)

        Returns:
            Normalized attack type string (e.g., "benign", "dos", "brute_force")
        """
        # Step 1: Handle CIC-IDS2018 numeric labels
        if isinstance(attack_type, (int, float)):
            label_num = int(attack_type)
            attack_type = self.CICIDS2018_LABEL_MAPPING.get(
                label_num,
                f"Unknown-{label_num}"
            )

        # Step 2: Normalize to standard categories
        attack_str = str(attack_type).strip()

        # Try CIC-IDS2018 specific mapping first
        if attack_str in self.CICIDS2018_TO_STANDARD:
            return self.CICIDS2018_TO_STANDARD[attack_str]

        # Fall back to CIC-IDS2017 mapping
        attack_lower = attack_str.lower()

        # CIC-IDS2017 text label mapping
        ids2017_mapping = {
            'normal traffic': 'benign',
            'dos': 'dos',
            'ddos': 'ddos',
            'port scanning': 'port_scan',
            'portscan': 'port_scan',
            'brute force': 'brute_force',
            'bruteforce': 'brute_force',
            'web attacks': 'web_attack',
            'web attack': 'web_attack',
            'bots': 'botnet',
            'bot': 'botnet',
            'infiltration': 'infiltration'
        }

        result = ids2017_mapping.get(attack_lower, attack_str.lower().replace(' ', '_'))

        return result


def stratified_sample_cicids(
    df: pd.DataFrame,
    n_samples: int = 50,
    attack_column: str = 'Attack Type',
    random_state: int = 42,
    dataset_version: str = '2017'
) -> pd.DataFrame:
    """
    Perform stratified sampling from CIC-IDS dataset

    Supports both CIC-IDS2017 (text labels) and CIC-IDS2018 (numeric labels).

    Target distribution for 50 samples (CIC-IDS2017):
    - Normal Traffic: 20 (40%)
    - DoS: 10 (20%)
    - DDoS: 8 (16%)
    - Port Scanning: 6 (12%)
    - Brute Force: 3 (6%)
    - Web Attacks: 2 (4%)
    - Bots: 1 (2%)

    Target distribution for 50 samples (CIC-IDS2018):
    - Label 1 (Benign): 25 (~50%)
    - Labels 2-10 (Attacks): 25 total (~50%, proportionally distributed)

    Args:
        df: CIC-IDS DataFrame
        n_samples: Total number of samples (default 50)
        attack_column: Name of the label column
            - CIC-IDS2017: 'Attack Type' (text labels)
            - CIC-IDS2018: 'Label' (numeric labels 1-10)
        random_state: Random seed for reproducibility
        dataset_version: '2017' or '2018' (default '2017')

    Returns:
        Stratified sample DataFrame

    Examples:
        # For CIC-IDS2017
        df_2017 = pd.read_csv('datasets/cicids2017/cicids2017_cleaned.csv')
        sample_2017 = stratified_sample_cicids(
            df_2017, n_samples=50, dataset_version='2017'
        )

        # For CIC-IDS2018
        df_2018 = pd.read_csv('datasets/cicids2018/cleaned_ids2018_sampled.csv')
        sample_2018 = stratified_sample_cicids(
            df_2018, n_samples=50, attack_column='Label', dataset_version='2018'
        )
    """
    # Define target distribution based on dataset version
    if dataset_version == '2018':
        # CIC-IDS2018: Work with numeric labels (1-10)
        # Based on actual distribution from dataset analysis
        # Target: ~50% benign (Label 1), ~50% attacks (Labels 2-10)
        target_distribution = {
            1: 25,   # Benign (77.57% in dataset) - 50% of sample
            4: 6,    # 2nd most common (10.67%) - frequent attack
            5: 4,    # 3rd (4.25%)
            2: 4,    # 4th (3.30%)
            3: 4,    # 5th (3.26%)
            6: 3,    # 6th (0.63%)
            7: 2,    # Rare (0.23%) - may have insufficient samples
            8: 1,    # Very rare (0.06%) - may have insufficient samples
            9: 1,    # Very rare (0.02%) - may have insufficient samples
            # Label 10 omitted (0.01%) - too rare for reliable sampling
        }
    else:
        # CIC-IDS2017: Original text-based distribution
        target_distribution = {
            'Normal Traffic': 20,
            'DoS': 10,
            'DDoS': 8,
            'Port Scanning': 6,
            'Brute Force': 3,
            'Web Attacks': 2,
            'Bots': 1
        }

    # Use the attack column directly (already has the right labels)
    df['Category'] = df[attack_column]

    # Sample from each category
    sampled_frames = []

    for category, count in target_distribution.items():
        category_df = df[df['Category'] == category]

        if len(category_df) == 0:
            print(f"Warning: No samples found for category '{category}'")
            continue

        if len(category_df) < count:
            print(f"Warning: Only {len(category_df)} samples available for '{category}', requested {count}")
            sample_count = len(category_df)
        else:
            sample_count = count

        sample = category_df.sample(n=sample_count, random_state=random_state)
        sampled_frames.append(sample)

    # Combine all samples
    result = pd.concat(sampled_frames, ignore_index=True)

    print(f"\nStratified sampling summary:")
    print(f"Dataset version: {dataset_version}")
    print(f"Total samples: {len(result)}")
    print(f"\nDistribution:")
    for category in result['Category'].value_counts().items():
        print(f"  {category[0]}: {category[1]} samples")

    return result


if __name__ == "__main__":
    # Example usage
    print("CIC-IDS2017 Feature-to-Text Converter")
    print("=" * 50)

    # Example flow data
    example_flow = {
        'Destination Port': 80,
        'Flow Duration': 2500000,  # microseconds
        'Total Fwd Packets': 125000,
        'Flow Bytes/s': 62500000,
        'Flow Packets/s': 50000,
        'Flow IAT Mean': 20,
        'FIN Flag Count': 0,
        'PSH Flag Count': 10,
        'ACK Flag Count': 0,
        'Average Packet Size': 500,
        'Attack Type': 'DoS'
    }

    converter = CICIDSConverter()
    description = converter.convert_flow_to_text(example_flow)

    print("\nExample DoS flow:")
    print(f"Description: {description}")

    # Example normal traffic
    normal_flow = {
        'Destination Port': 443,
        'Flow Duration': 45000000,
        'Total Fwd Packets': 150,
        'Flow Bytes/s': 150000,
        'Flow Packets/s': 3.3,
        'Flow IAT Mean': 300000,
        'FIN Flag Count': 1,
        'PSH Flag Count': 50,
        'ACK Flag Count': 100,
        'Average Packet Size': 1024,
        'Attack Type': 'Normal Traffic'
    }

    description = converter.convert_flow_to_text(normal_flow)
    print("\nExample normal HTTPS flow:")
    print(f"Description: {description}")
