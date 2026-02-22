"""
PROBLEM 1 SOLUTION: Network Traffic Analysis
Analyzing traffic to identify patterns that may indicate attacks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
print("="*80)
print("PROBLEM 1: NETWORK TRAFFIC ANALYSIS")
print("="*80)
print("\nLoading dataset...")

df = pd.read_csv('dataset1_network_traffic.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Dataset loaded successfully!")
print(f"Total records: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nFirst few records:")
print(df.head())

# Data exploration
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn data types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nBasic statistics:")
print(df.describe())

# Create output directory for plots
import os
os.makedirs('plots_problem1', exist_ok=True)

# ============================================================================
# Q1: TRAFFIC VOLUME ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("Q1: TRAFFIC VOLUME ANALYSIS")
print("="*80)

# Q1a: Peak traffic hours
print("\nQ1a: Peak Traffic Hours")
hourly_traffic = df.groupby('hour').agg({
    'bytes': 'sum',
    'packets': 'sum',
    'timestamp': 'count'
}).rename(columns={'timestamp': 'connection_count'})

hourly_traffic['bytes_mb'] = hourly_traffic['bytes'] / (1024 * 1024)

print("\nTraffic by Hour:")
print(hourly_traffic.sort_values('bytes', ascending=False))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Traffic volume by hour
axes[0].bar(hourly_traffic.index, hourly_traffic['bytes_mb'], color='steelblue', alpha=0.7)
axes[0].set_xlabel('Hour of Day', fontsize=12)
axes[0].set_ylabel('Total Data (MB)', fontsize=12)
axes[0].set_title('Network Traffic Volume by Hour of Day', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Connection count by hour
axes[1].plot(hourly_traffic.index, hourly_traffic['connection_count'], 
             marker='o', linewidth=2, markersize=6, color='coral')
axes[1].set_xlabel('Hour of Day', fontsize=12)
axes[1].set_ylabel('Number of Connections', fontsize=12)
axes[1].set_title('Connection Count by Hour of Day', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_problem1/q1a_hourly_traffic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q1a_hourly_traffic.png")

# Q1b: Traffic by day of week
print("\nQ1b: Traffic by Day of Week")
daily_traffic = df.groupby('day_of_week').agg({
    'bytes': 'sum',
    'packets': 'sum',
    'timestamp': 'count'
}).rename(columns={'timestamp': 'connection_count'})

daily_traffic['bytes_gb'] = daily_traffic['bytes'] / (1024 * 1024 * 1024)

# Order days properly
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_traffic = daily_traffic.reindex([d for d in day_order if d in daily_traffic.index])

print("\nTraffic by Day of Week:")
print(daily_traffic)

plt.figure(figsize=(12, 6))
plt.bar(daily_traffic.index, daily_traffic['bytes_gb'], color='teal', alpha=0.7)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Data (GB)', fontsize=12)
plt.title('Network Traffic Volume by Day of Week', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots_problem1/q1b_daily_traffic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q1b_daily_traffic.png")

# Q1c: Traffic spikes detection
print("\nQ1c: Traffic Spikes Detection")
df['date'] = df['timestamp'].dt.date
daily_bytes = df.groupby('date')['bytes'].sum() / (1024 * 1024)  # Convert to MB

# Calculate rolling mean and std for anomaly detection
rolling_mean = daily_bytes.rolling(window=3, center=True).mean()
rolling_std = daily_bytes.rolling(window=3, center=True).std()
threshold = rolling_mean + (2 * rolling_std)

spikes = daily_bytes[daily_bytes > threshold]
print(f"\nDetected {len(spikes)} traffic spikes:")
print(spikes)

plt.figure(figsize=(14, 6))
plt.plot(daily_bytes.index, daily_bytes.values, marker='o', linewidth=2, label='Daily Traffic')
plt.plot(rolling_mean.index, rolling_mean.values, '--', linewidth=2, label='Rolling Mean', alpha=0.7)
plt.plot(threshold.index, threshold.values, ':', linewidth=2, label='Spike Threshold', alpha=0.7)
plt.scatter(spikes.index, spikes.values, color='red', s=100, zorder=5, label='Spikes', marker='^')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Data (MB)', fontsize=12)
plt.title('Daily Traffic with Spike Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots_problem1/q1c_traffic_spikes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q1c_traffic_spikes.png")

# ============================================================================
# Q2: PROTOCOL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("Q2: PROTOCOL ANALYSIS")
print("="*80)

# Q2a: Protocol distribution
print("\nQ2a: Protocol Distribution")
protocol_stats = df.groupby('protocol').agg({
    'bytes': 'sum',
    'packets': 'sum',
    'timestamp': 'count'
}).rename(columns={'timestamp': 'connection_count'})

protocol_stats['bytes_mb'] = protocol_stats['bytes'] / (1024 * 1024)
protocol_stats = protocol_stats.sort_values('bytes', ascending=False)

print("\nProtocol Statistics:")
print(protocol_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart for connection count
axes[0].pie(protocol_stats['connection_count'], labels=protocol_stats.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette('Set2'))
axes[0].set_title('Protocol Distribution by Connection Count', fontsize=14, fontweight='bold')

# Bar chart for bandwidth
axes[1].barh(protocol_stats.index, protocol_stats['bytes_mb'], color='skyblue', alpha=0.7)
axes[1].set_xlabel('Total Data (MB)', fontsize=12)
axes[1].set_ylabel('Protocol', fontsize=12)
axes[1].set_title('Bandwidth Consumption by Protocol', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots_problem1/q2a_protocol_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q2a_protocol_distribution.png")

# Q2c: Unusual protocol patterns
print("\nQ2c: Unusual Protocol Patterns")
# Look for protocols with abnormally high bytes per connection
protocol_stats['avg_bytes_per_conn'] = protocol_stats['bytes'] / protocol_stats['connection_count']
print("\nAverage bytes per connection by protocol:")
print(protocol_stats[['connection_count', 'bytes_mb', 'avg_bytes_per_conn']].sort_values('avg_bytes_per_conn', ascending=False))

# ============================================================================
# Q3: PORT SCAN DETECTION
# ============================================================================
print("\n" + "="*80)
print("Q3: PORT SCAN DETECTION")
print("="*80)

# Q3a: Identify port scanners
print("\nQ3a: Identifying Port Scanners (IPs connecting to >20 ports)")
port_scan_analysis = df.groupby('source_ip').agg({
    'destination_port': 'nunique',
    'destination_ip': 'nunique',
    'timestamp': 'count'
}).rename(columns={
    'destination_port': 'unique_ports',
    'destination_ip': 'unique_targets',
    'timestamp': 'connection_count'
})

port_scanners = port_scan_analysis[port_scan_analysis['unique_ports'] > 20].sort_values('unique_ports', ascending=False)

print(f"\nFound {len(port_scanners)} potential port scanners:")
print(port_scanners.head(10))

# Q3b: Port scanning over time
print("\nQ3b: Port Scanning Activity Over Time")
if len(port_scanners) > 0:
    scanner_ips = port_scanners.head(5).index.tolist()
    scan_timeline = df[df['source_ip'].isin(scanner_ips)].copy()
    scan_timeline['date'] = scan_timeline['timestamp'].dt.date
    
    scan_daily = scan_timeline.groupby(['date', 'source_ip']).size().reset_index(name='scan_count')
    
    plt.figure(figsize=(14, 6))
    for ip in scanner_ips:
        ip_data = scan_daily[scan_daily['source_ip'] == ip]
        plt.plot(ip_data['date'], ip_data['scan_count'], marker='o', linewidth=2, label=ip)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Connection Attempts', fontsize=12)
    plt.title('Port Scanning Activity Timeline (Top 5 Scanners)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots_problem1/q3b_port_scan_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots_problem1/q3b_port_scan_timeline.png")

# Q3c: Top port scanners visualization
print("\nQ3c: Top Port Scanners")
plt.figure(figsize=(12, 6))
top_scanners = port_scanners.head(10)
x = range(len(top_scanners))
width = 0.35

plt.bar([i - width/2 for i in x], top_scanners['unique_ports'], width, label='Unique Ports', alpha=0.8)
plt.bar([i + width/2 for i in x], top_scanners['unique_targets'], width, label='Unique Targets', alpha=0.8)

plt.xlabel('Source IP', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Top 10 Port Scanners: Ports and Targets Scanned', fontsize=14, fontweight='bold')
plt.xticks(x, top_scanners.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots_problem1/q3c_top_port_scanners.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q3c_top_port_scanners.png")

# ============================================================================
# Q4: DDOS ATTACK DETECTION
# ============================================================================
print("\n" + "="*80)
print("Q4: DDOS ATTACK DETECTION")
print("="*80)

# Q4a: High packet count connections
print("\nQ4a: Connections with High Packet Counts (>100 packets)")
high_packet_conns = df[df['packets'] > 100].copy()
print(f"Found {len(high_packet_conns)} high-packet connections")

print("\nTop 10 connections by packet count:")
print(high_packet_conns.nlargest(10, 'packets')[['timestamp', 'source_ip', 'destination_ip', 
                                                   'packets', 'bytes', 'protocol']])

# Q4b: Identify DDoS targets
print("\nQ4b: Identifying Potential DDoS Targets")
ddos_targets = high_packet_conns.groupby('destination_ip').agg({
    'source_ip': 'nunique',
    'packets': 'sum',
    'bytes': 'sum',
    'timestamp': 'count'
}).rename(columns={
    'source_ip': 'unique_sources',
    'timestamp': 'connection_count'
})

ddos_targets = ddos_targets[ddos_targets['unique_sources'] > 5].sort_values('packets', ascending=False)
print(f"\nFound {len(ddos_targets)} potential DDoS targets:")
print(ddos_targets.head(10))

# Q4c: DDoS attack timeline
print("\nQ4c: DDoS Attack Timeline")
if len(ddos_targets) > 0:
    target_ips = ddos_targets.head(3).index.tolist()
    ddos_timeline = high_packet_conns[high_packet_conns['destination_ip'].isin(target_ips)].copy()
    ddos_timeline['hour'] = ddos_timeline['timestamp'].dt.floor('H')
    
    ddos_hourly = ddos_timeline.groupby(['hour', 'destination_ip']).agg({
        'packets': 'sum',
        'source_ip': 'nunique'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for ip in target_ips:
        ip_data = ddos_hourly[ddos_hourly['destination_ip'] == ip]
        axes[0].plot(ip_data['hour'], ip_data['packets'], marker='o', linewidth=2, label=ip)
        axes[1].plot(ip_data['hour'], ip_data['source_ip'], marker='s', linewidth=2, label=ip)
    
    axes[0].set_ylabel('Total Packets', fontsize=12)
    axes[0].set_title('Potential DDoS: Packet Volume Over Time', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Unique Source IPs', fontsize=12)
    axes[1].set_title('Potential DDoS: Number of Attacking Sources', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots_problem1/q4c_ddos_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots_problem1/q4c_ddos_timeline.png")

# ============================================================================
# Q5: DATA EXFILTRATION DETECTION
# ============================================================================
print("\n" + "="*80)
print("Q5: DATA EXFILTRATION DETECTION")
print("="*80)

# Q5a: High data transfers
print("\nQ5a: Connections with High Data Transfer (>500 KB)")
threshold_bytes = 500 * 1024  # 500 KB
high_data_conns = df[df['bytes'] > threshold_bytes].copy()
print(f"Found {len(high_data_conns)} high-data connections")

print("\nTop 10 data transfers:")
print(high_data_conns.nlargest(10, 'bytes')[['timestamp', 'source_ip', 'destination_ip', 
                                               'bytes', 'duration_seconds', 'protocol']])

# Q5b: Internal to external large transfers
print("\nQ5b: Internal IPs Sending Large Data to External IPs")
high_data_conns['src_is_internal'] = high_data_conns['source_ip'].str.startswith('192.168.')
high_data_conns['dst_is_internal'] = high_data_conns['destination_ip'].str.startswith('192.168.')

exfiltration_suspects = high_data_conns[
    (high_data_conns['src_is_internal'] == True) & 
    (high_data_conns['dst_is_internal'] == False)
]

print(f"\nFound {len(exfiltration_suspects)} potential data exfiltration events")

exfil_summary = exfiltration_suspects.groupby('source_ip').agg({
    'bytes': ['sum', 'count'],
    'destination_ip': 'nunique'
}).round(2)

exfil_summary.columns = ['total_bytes', 'transfer_count', 'unique_destinations']
exfil_summary['total_mb'] = exfil_summary['total_bytes'] / (1024 * 1024)
exfil_summary = exfil_summary.sort_values('total_bytes', ascending=False)

print("\nTop internal IPs by data sent externally:")
print(exfil_summary.head(10))

# Q5c: Visualization of data exfiltration
print("\nQ5c: Data Exfiltration Patterns")
if len(exfil_summary) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top sources
    top_sources = exfil_summary.head(10)
    axes[0].barh(range(len(top_sources)), top_sources['total_mb'], color='crimson', alpha=0.7)
    axes[0].set_yticks(range(len(top_sources)))
    axes[0].set_yticklabels(top_sources.index)
    axes[0].set_xlabel('Total Data Sent (MB)', fontsize=12)
    axes[0].set_ylabel('Source IP (Internal)', fontsize=12)
    axes[0].set_title('Top 10 Internal IPs by External Data Transfer', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Scatter plot: transfer count vs data volume
    axes[1].scatter(exfil_summary['transfer_count'], exfil_summary['total_mb'], 
                   s=exfil_summary['unique_destinations']*20, alpha=0.6, color='darkred')
    axes[1].set_xlabel('Number of Transfers', fontsize=12)
    axes[1].set_ylabel('Total Data Sent (MB)', fontsize=12)
    axes[1].set_title('Data Exfiltration Pattern Analysis\n(Bubble size = unique destinations)', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots_problem1/q5c_data_exfiltration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots_problem1/q5c_data_exfiltration.png")

# ============================================================================
# Q6: BRUTE FORCE ATTACK DETECTION
# ============================================================================
print("\n" + "="*80)
print("Q6: BRUTE FORCE ATTACK DETECTION")
print("="*80)

# Q6a & Q6b: Repeated authentication attempts
print("\nQ6a & Q6b: Detecting Brute Force Attempts on SSH/RDP")
auth_ports = [22, 3389]  # SSH and RDP
auth_attempts = df[df['destination_port'].isin(auth_ports)].copy()

print(f"Total authentication-related connections: {len(auth_attempts)}")

brute_force = auth_attempts.groupby(['source_ip', 'destination_ip', 'destination_port']).agg({
    'timestamp': ['count', 'min', 'max']
}).reset_index()

brute_force.columns = ['source_ip', 'destination_ip', 'port', 'attempt_count', 'first_attempt', 'last_attempt']
brute_force['duration'] = (pd.to_datetime(brute_force['last_attempt']) - 
                           pd.to_datetime(brute_force['first_attempt'])).dt.total_seconds() / 60  # minutes

brute_force_suspects = brute_force[brute_force['attempt_count'] > 10].sort_values('attempt_count', ascending=False)

print(f"\nFound {len(brute_force_suspects)} potential brute force attacks:")
print(brute_force_suspects.head(10))

# Q6c: Heatmap of brute force activity
print("\nQ6c: Brute Force Activity Heatmap")
if len(auth_attempts) > 0:
    auth_attempts['date'] = auth_attempts['timestamp'].dt.date
    auth_attempts['hour'] = auth_attempts['timestamp'].dt.hour
    
    heatmap_data = auth_attempts.groupby(['date', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='date', values='count').fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Attempt Count'})
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Hour of Day', fontsize=12)
    plt.title('Brute Force Attack Attempts Heatmap (SSH/RDP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots_problem1/q6c_bruteforce_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots_problem1/q6c_bruteforce_heatmap.png")

# ============================================================================
# Q7: TOP TALKERS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("Q7: TOP TALKERS ANALYSIS")
print("="*80)

# Q7a: Top source IPs
print("\nQ7a: Top 10 Most Active Source IPs")
top_sources = df.groupby('source_ip').agg({
    'bytes': 'sum',
    'packets': 'sum',
    'timestamp': 'count',
    'destination_ip': 'nunique',
    'destination_port': 'nunique'
}).rename(columns={'timestamp': 'connections', 'destination_ip': 'unique_targets', 
                   'destination_port': 'unique_ports'})

top_sources['bytes_mb'] = top_sources['bytes'] / (1024 * 1024)
top_sources = top_sources.sort_values('bytes', ascending=False)

print("\nTop 10 Source IPs by Traffic Volume:")
print(top_sources.head(10))

# Q7b: Top destination IPs
print("\nQ7b: Top 10 Most Targeted Destination IPs")
top_dests = df.groupby('destination_ip').agg({
    'bytes': 'sum',
    'packets': 'sum',
    'timestamp': 'count',
    'source_ip': 'nunique',
    'destination_port': 'nunique'
}).rename(columns={'timestamp': 'connections', 'source_ip': 'unique_sources', 
                   'destination_port': 'unique_ports'})

top_dests['bytes_mb'] = top_dests['bytes'] / (1024 * 1024)
top_dests = top_dests.sort_values('bytes', ascending=False)

print("\nTop 10 Destination IPs by Traffic Volume:")
print(top_dests.head(10))

# Q7c: Suspicious behavior identification
print("\nQ7c: Identifying Suspicious Top Talkers")

# Flag suspicious sources
suspicious_criteria = (
    (top_sources['unique_ports'] > 20) |  # Port scanning
    (top_sources['unique_targets'] > 50) |  # Many targets
    (top_sources['connections'] > 100)  # High connection count
)
suspicious_sources = top_sources[suspicious_criteria]

print(f"\n{len(suspicious_sources)} suspicious source IPs identified:")
print(suspicious_sources.head(10))

# Flag suspicious destinations
suspicious_dest_criteria = (
    (top_dests['unique_sources'] > 20) |  # Many sources (potential DDoS target)
    (top_dests['connections'] > 200)  # Very high connection count
)
suspicious_dests = top_dests[suspicious_dest_criteria]

print(f"\n{len(suspicious_dests)} suspicious destination IPs identified:")
print(suspicious_dests.head(10))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top sources by volume
top10_src = top_sources.head(10)
axes[0, 0].barh(range(len(top10_src)), top10_src['bytes_mb'], color='steelblue', alpha=0.7)
axes[0, 0].set_yticks(range(len(top10_src)))
axes[0, 0].set_yticklabels(top10_src.index)
axes[0, 0].set_xlabel('Total Data (MB)', fontsize=11)
axes[0, 0].set_title('Top 10 Source IPs by Traffic Volume', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Top destinations by volume
top10_dst = top_dests.head(10)
axes[0, 1].barh(range(len(top10_dst)), top10_dst['bytes_mb'], color='coral', alpha=0.7)
axes[0, 1].set_yticks(range(len(top10_dst)))
axes[0, 1].set_yticklabels(top10_dst.index)
axes[0, 1].set_xlabel('Total Data (MB)', fontsize=11)
axes[0, 1].set_title('Top 10 Destination IPs by Traffic Volume', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Suspicious source behavior
if len(suspicious_sources) > 0:
    susp_src_plot = suspicious_sources.head(10)
    axes[1, 0].scatter(susp_src_plot['unique_ports'], susp_src_plot['unique_targets'],
                      s=susp_src_plot['connections'], alpha=0.6, color='red')
    axes[1, 0].set_xlabel('Unique Ports Accessed', fontsize=11)
    axes[1, 0].set_ylabel('Unique Target IPs', fontsize=11)
    axes[1, 0].set_title('Suspicious Source IPs Behavior\n(Bubble size = connection count)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

# Suspicious destination behavior
if len(suspicious_dests) > 0:
    susp_dst_plot = suspicious_dests.head(10)
    axes[1, 1].scatter(susp_dst_plot['unique_sources'], susp_dst_plot['connections'],
                      s=susp_dst_plot['bytes_mb']*0.1, alpha=0.6, color='darkred')
    axes[1, 1].set_xlabel('Unique Source IPs', fontsize=11)
    axes[1, 1].set_ylabel('Connection Count', fontsize=11)
    axes[1, 1].set_title('Suspicious Destination IPs\n(Bubble size = traffic volume)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots_problem1/q7_top_talkers.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots_problem1/q7_top_talkers.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\n📊 KEY FINDINGS:")
print(f"\n1. TRAFFIC PATTERNS:")
print(f"   • Peak traffic hour: {hourly_traffic['bytes'].idxmax()}:00")
print(f"   • Busiest day: {daily_traffic['bytes'].idxmax()}")
print(f"   • Detected {len(spikes)} traffic spikes")

print(f"\n2. POTENTIAL ATTACKS DETECTED:")
print(f"   • Port Scanners: {len(port_scanners)} IPs")
print(f"   • DDoS Targets: {len(ddos_targets)} IPs under attack")
print(f"   • Data Exfiltration: {len(exfil_summary)} internal IPs")
print(f"   • Brute Force Attempts: {len(brute_force_suspects)} incidents")

print(f"\n3. PROTOCOL ANALYSIS:")
print(f"   • Most used protocol: {protocol_stats['connection_count'].idxmax()}")
print(f"   • Highest bandwidth: {protocol_stats['bytes'].idxmax()}")

print(f"\n4. TOP TALKERS:")
print(f"   • Suspicious sources: {len(suspicious_sources)}")
print(f"   • Suspicious destinations: {len(suspicious_dests)}")

print("\n✅ All visualizations saved in 'plots_problem1/' directory")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
