import gradio as gr
import subprocess
import re
from datetime import datetime
from collections import defaultdict

# GPU memory mapping (in GB)
GPU_MEMORY = {
    'a100': 40,      # A100 (也有80GB版本，但默认40GB)
    'h200': 141,     # H200
    'v100': 16,      # V100 (也有32GB版本，但默认16GB)
    'a5000': 24,     # A5000
    'a5500': 24,     # A5500
    'l40s': 48,      # L40S
    'l40sx': 48,     # L40S variant
    '2080ti': 11,    # RTX 2080 Ti
    'rtx2080ti': 11,
    'a6000': 48,     # A6000
    'a40': 48,       # A40
    'rtx3090': 24,   # RTX 3090
    'rtx4090': 24,   # RTX 4090
    'titan': 24,     # Titan RTX
}

def get_detailed_partition_info():
    """Get detailed partition information including resource allocation"""
    try:
        # Get detailed partition info with CPU allocation
        result = subprocess.run(
            ['sinfo', '-o', '%P %F %C %D %l %T %N'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return None, "Error running sinfo command"
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None, "No sinfo output"
        
        partitions = {}
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 7:
                partition_name = parts[0].replace('*', '')
                
                # Parse nodes info: allocated/idle/other/total
                node_info = parts[1].split('/')
                if len(node_info) == 4:
                    allocated_nodes = int(node_info[0])
                    idle_nodes = int(node_info[1])
                    other_nodes = int(node_info[2])
                    total_nodes = int(node_info[3])
                else:
                    continue
                
                # Parse CPU info: allocated/idle/other/total
                cpu_info = parts[2].split('/')
                if len(cpu_info) == 4:
                    allocated_cpus = int(cpu_info[0])
                    idle_cpus = int(cpu_info[1])
                    other_cpus = int(cpu_info[2])
                    total_cpus = int(cpu_info[3])
                else:
                    continue
                
                nodes_count = int(parts[3])
                timelimit = parts[4]
                state = parts[5]
                nodelist = ' '.join(parts[6:])
                
                # Aggregate by partition name (handle multiple lines for same partition)
                if partition_name not in partitions:
                    partitions[partition_name] = {
                        'name': partition_name,
                        'allocated_nodes': 0,
                        'idle_nodes': 0,
                        'other_nodes': 0,
                        'total_nodes': 0,
                        'allocated_cpus': 0,
                        'idle_cpus': 0,
                        'other_cpus': 0,
                        'total_cpus': 0,
                        'timelimit': timelimit,
                        'states': [],
                        'nodelists': []
                    }
                
                partitions[partition_name]['allocated_nodes'] += allocated_nodes
                partitions[partition_name]['idle_nodes'] += idle_nodes
                partitions[partition_name]['other_nodes'] += other_nodes
                partitions[partition_name]['total_nodes'] += total_nodes
                partitions[partition_name]['allocated_cpus'] += allocated_cpus
                partitions[partition_name]['idle_cpus'] += idle_cpus
                partitions[partition_name]['other_cpus'] += other_cpus
                partitions[partition_name]['total_cpus'] += total_cpus
                partitions[partition_name]['states'].append(state)
                partitions[partition_name]['nodelists'].append(nodelist)
        
        return list(partitions.values()), None
        
    except subprocess.TimeoutExpired:
        return None, "sinfo command timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"

def infer_gpu_memory(partition_name):
    """Infer GPU memory from partition name"""
    partition_lower = partition_name.lower()
    
    for gpu_type, memory in GPU_MEMORY.items():
        if gpu_type in partition_lower:
            return memory
    
    return None

def create_progress_bar(used, total, color_scheme='green'):
    """Create an HTML progress bar"""
    if total == 0:
        percentage = 0
    else:
        percentage = (used / total) * 100
    
    available = total - used
    
    # Color schemes based on availability
    if percentage >= 90:
        bar_color = '#EF5350'  # Red
        bg_color = '#FFCDD2'
    elif percentage >= 70:
        bar_color = '#FFA726'  # Orange
        bg_color = '#FFE0B2'
    elif percentage >= 40:
        bar_color = '#FFA726'  # Orange
        bg_color = '#FFE0B2'
    else:
        bar_color = '#66BB6A'  # Green
        bg_color = '#C8E6C9'
    
    html = f"""
    <div style='width: 100%; background: {bg_color}; border-radius: 8px; overflow: hidden; height: 24px; position: relative;'>
        <div style='width: {percentage}%; background: {bar_color}; height: 100%; transition: width 0.3s ease;'></div>
        <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: 600; font-size: 12px; color: #333;'>
            {used}/{total} used ({available} available)
        </div>
    </div>
    """
    return html

def get_available_resources(min_gpu_memory=0):
    """Get available GPU/CPU resources with detailed progress bars"""
    partitions, error = get_detailed_partition_info()
    
    if error:
        return f"❌ **Error:** {error}"
    
    # Build resource summary with better formatting
    summary = []
    summary.append(f"<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>")
    summary.append(f"<h2 style='margin: 0; font-size: 24px;'>📊 Cluster Resource Status</h2>")
    summary.append(f"<p style='margin: 10px 0 0 0; opacity: 0.9;'>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    summary.append("</div>")
    summary.append("")
    
    # Categorize and sort partitions by GPU type
    gpu_partitions = defaultdict(list)
    cpu_partitions = []
    filtered_results = []  # Initialize here to avoid UnboundLocalError
    
    for partition in partitions:
        name = partition['name']
        gpu_mem = infer_gpu_memory(name)
        
        # Calculate availability score for sorting
        total_nodes = partition['total_nodes']
        available_nodes = partition['idle_nodes'] + (partition['total_nodes'] - partition['allocated_nodes'] - partition['idle_nodes'] - partition['other_nodes'])
        availability_score = available_nodes if total_nodes > 0 else 0
        
        partition['availability_score'] = availability_score
        partition['gpu_memory'] = gpu_mem
        
        # if gpu_mem:
        #     if min_gpu_memory == 0 or gpu_mem >= min_gpu_memory:
        #         gpu_partitions[gpu_mem].append(partition)
        # else:
        #     cpu_partitions.append(partition)
                # Compute available nodes robustly (nodes not allocated)
        # Use max(0, ...) to avoid negative numbers if data is inconsistent
        total_nodes = partition.get('total_nodes', 0)
        allocated_nodes = partition.get('allocated_nodes', 0)
        other_nodes = partition.get('other_nodes', 0)
        # available_nodes = total - allocated (we treat 'other' as not available)
        available_nodes = max(0, total_nodes - allocated_nodes - other_nodes)

        # store available_nodes for later use and correct availability score
        partition['available_nodes'] = available_nodes
        availability_score = available_nodes if total_nodes > 0 else 0
        partition['availability_score'] = availability_score
        partition['gpu_memory'] = gpu_mem

        # IMPORTANT FIX:
        # Don't filter GPU types here using per-card memory vs min_gpu_memory.
        # We should keep all GPU types and apply the total-available check per partition later.
        if gpu_mem:
            gpu_partitions[gpu_mem].append(partition)
        else:
            cpu_partitions.append(partition)

    
    # Sort each GPU memory category by availability (highest first)
    for gpu_mem in gpu_partitions:
        gpu_partitions[gpu_mem].sort(key=lambda x: (x['availability_score'], x['idle_nodes']), reverse=True)
    
    # Sort CPU partitions by availability
    cpu_partitions.sort(key=lambda x: (x['availability_score'], x['idle_nodes']), reverse=True)
    
    # GPU Resources Section
    if gpu_partitions:
        summary.append("<div style='margin: 20px 0;'>")
        if min_gpu_memory > 0:
            summary.append(f"<h3 style='color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px;'>🎮 GPU Partitions with ≥ {min_gpu_memory}GB Total Available Memory</h3>")
            summary.append(f"<p style='color: #666; font-size: 14px; margin-top: 10px;'>💡 Filtering by <strong>Total Available GPU Memory</strong> = GPU Memory per Card × Available Nodes (per partition)</p>")
        else:
            summary.append("<h3 style='color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px;'>🎮 All GPU Resources (Sorted by Availability)</h3>")
        summary.append("</div>")
        
        has_displayed_gpu = False  # Track if we displayed any GPU
        displayed_gpu_types = 0  # Count actually displayed GPU types
        
        for gpu_mem in sorted(gpu_partitions.keys(), reverse=True):
            partitions_list = gpu_partitions[gpu_mem]
            
            # Filter partitions at the individual partition level, not GPU type level
            filtered_partitions = []
            for partition in partitions_list:
                partition_available_nodes = partition['total_nodes'] - partition['allocated_nodes']
                partition_total_gpu_memory = gpu_mem * partition_available_nodes
                
                # Only include partitions that meet the memory requirement
                if min_gpu_memory == 0 or partition_total_gpu_memory >= min_gpu_memory:
                    filtered_partitions.append(partition)
            
            # Skip this GPU type if no partitions pass the filter
            if not filtered_partitions:
                continue
            
            has_displayed_gpu = True  # Mark that we displayed at least one GPU type
            displayed_gpu_types += 1  # Count this GPU type
            
            # Calculate statistics only for filtered partitions
            total_nodes_sum = sum(p['total_nodes'] for p in filtered_partitions)
            idle_nodes_sum = sum(p['idle_nodes'] for p in filtered_partitions)
            allocated_nodes_sum = sum(p['allocated_nodes'] for p in filtered_partitions)
            available_nodes_sum = total_nodes_sum - allocated_nodes_sum
            
            # Calculate total available GPU memory for filtered partitions
            total_available_gpu_memory = gpu_mem * available_nodes_sum
            
            # GPU Memory Card Header with overall stats
            if available_nodes_sum > total_nodes_sum * 0.3:
                card_color = "#4CAF50"  # Green
            elif available_nodes_sum > 0:
                card_color = "#FFA726"  # Orange
            else:
                card_color = "#EF5350"  # Red
            
            summary.append(f"<div style='background: {card_color}; color: white; padding: 15px; border-radius: 10px 10px 0 0; margin-top: 20px;'>")
            if min_gpu_memory > 0 and len(filtered_partitions) < len(partitions_list):
                summary.append(f"<h4 style='margin: 0; font-size: 20px;'>💾 {gpu_mem}GB per GPU | Showing {len(filtered_partitions)} of {len(partitions_list)} partitions</h4>")
                summary.append(f"<p style='margin: 5px 0 0 0; opacity: 0.9;'>Total Available in Shown Partitions: {total_available_gpu_memory}GB | {available_nodes_sum} available nodes, {idle_nodes_sum} idle</p>")
            else:
                summary.append(f"<h4 style='margin: 0; font-size: 20px;'>💾 {gpu_mem}GB per GPU | Total Available: {total_available_gpu_memory}GB</h4>")
                summary.append(f"<p style='margin: 5px 0 0 0; opacity: 0.9;'>Available Nodes: {available_nodes_sum} | Idle Nodes: {idle_nodes_sum} | Total Nodes: {total_nodes_sum}</p>")
            summary.append("</div>")
            
            # Partition details with progress bars
            summary.append("<div style='background: #f8f9fa; padding: 15px; border-radius: 0 0 10px 10px; margin-bottom: 10px;'>")
            
            for partition in filtered_partitions:
                # Determine primary state
                dominant_state = 'idle' if partition['idle_nodes'] > partition['allocated_nodes'] else 'mix' if partition['idle_nodes'] > 0 else 'alloc'
                
                state_info = {
                    'mix': {'emoji': '🟡', 'color': '#FFA726', 'text': 'Partially Available'},
                    'alloc': {'emoji': '🔴', 'color': '#EF5350', 'text': 'Heavily Used'},
                    'idle': {'emoji': '🟢', 'color': '#66BB6A', 'text': 'Available'},
                    'down': {'emoji': '⚫', 'color': '#757575', 'text': 'Unavailable'}
                }.get(dominant_state, {'emoji': '⚪', 'color': '#BDBDBD', 'text': 'Unknown'})
                
                summary.append(
                    f"<div style='padding: 15px; margin: 10px 0; background: white; border-left: 4px solid {state_info['color']}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
                )
                
                # Partition header
                partition_available_nodes = partition['total_nodes'] - partition['allocated_nodes']
                partition_available_gpu_memory = gpu_mem * partition_available_nodes
                
                summary.append(
                    f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>"
                    f"<div>"
                    f"<strong style='font-size: 18px;'>{state_info['emoji']} {partition['name']}</strong> "
                    f"<span style='color: {state_info['color']}; font-weight: 600; font-size: 14px;'>{state_info['text']}</span>"
                    f"<br><span style='color: #667eea; font-weight: 600; font-size: 13px;'>🎮 Available GPU Memory: {partition_available_gpu_memory}GB ({gpu_mem}GB × {partition_available_nodes} nodes)</span>"
                    f"</div>"
                    f"<div style='color: #666; font-size: 14px;'>⏱️ Time Limit: {partition['timelimit']}</div>"
                    f"</div>"
                )
                
                # Node usage progress bar
                summary.append("<div style='margin: 10px 0;'>")
                summary.append("<div style='font-size: 13px; color: #666; margin-bottom: 5px; font-weight: 600;'>📦 Node Usage:</div>")
                summary.append(create_progress_bar(
                    partition['allocated_nodes'],
                    partition['total_nodes']
                ))
                summary.append("</div>")
                
                # CPU usage progress bar
                summary.append("<div style='margin: 10px 0;'>")
                summary.append("<div style='font-size: 13px; color: #666; margin-bottom: 5px; font-weight: 600;'>🔧 CPU Usage:</div>")
                summary.append(create_progress_bar(
                    partition['allocated_cpus'],
                    partition['total_cpus']
                ))
                summary.append("</div>")
                
                # Detailed stats
                summary.append(
                    f"<div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; font-size: 13px; color: #666;'>"
                    f"<strong>Details:</strong> "
                    f"Idle Nodes: {partition['idle_nodes']} | "
                    f"Allocated Nodes: {partition['allocated_nodes']} | "
                    f"Total CPUs: {partition['total_cpus']} | "
                    f"Available CPUs: {partition['idle_cpus'] + (partition['total_cpus'] - partition['allocated_cpus'] - partition['idle_cpus'] - partition['other_cpus'])}"
                    f"</div>"
                )
                
                summary.append("</div>")
                
                # Add to filtered results if available
                if partition['idle_nodes'] > 0 or (partition['total_nodes'] - partition['allocated_nodes']) > 0:
                    available_nodes = partition['total_nodes'] - partition['allocated_nodes']
                    total_gpu_memory = gpu_mem * available_nodes
                    filtered_results.append({
                        'partition': partition['name'],
                        'gpu_memory': gpu_mem,
                        'total_nodes': partition['total_nodes'],
                        'available_nodes': available_nodes,
                        'idle_nodes': partition['idle_nodes'],
                        'allocated_nodes': partition['allocated_nodes'],
                        'timelimit': partition['timelimit'],
                        'total_available_gpu_memory': total_gpu_memory,
                        'availability_pct': (available_nodes / partition['total_nodes'] * 100) if partition['total_nodes'] > 0 else 0
                    })
            
            summary.append("</div>")
        
        # Show statistics after the loop
        if min_gpu_memory > 0 and has_displayed_gpu:
            total_gpu_types = len(gpu_partitions)
            summary.append(f"<div style='margin: 20px 0; padding: 15px; background: #E3F2FD; border-radius: 8px; border-left: 4px solid #2196F3;'>")
            summary.append(f"<p style='margin: 0; color: #1976D2; font-size: 14px;'>✅ Showing <strong>{displayed_gpu_types} of {total_gpu_types}</strong> GPU types with partitions meeting the ≥{min_gpu_memory}GB requirement</p>")
            summary.append("</div>")
        
        # If no GPUs were displayed due to filtering, show a message
        if not has_displayed_gpu and min_gpu_memory > 0:
            summary.append("<div style='padding: 30px; background: #FFF3E0; border-radius: 10px; border: 2px dashed #FF9800; text-align: center;'>")
            summary.append(f"<h4 style='color: #F57C00; margin-top: 0;'>🔍 No GPU Partitions Found</h4>")
            summary.append(f"<p style='color: #666;'>No GPU partitions have ≥{min_gpu_memory}GB of <strong>total available GPU memory</strong>.</p>")
            summary.append(f"<p style='color: #666;'><strong>Total Available GPU Memory</strong> = GPU Memory per Card × Available Nodes</p>")
            summary.append("<p style='color: #666;'>Try lowering the memory filter or check back later when more nodes are available.</p>")
            summary.append("</div>")
    
    # CPU Resources Section
    if cpu_partitions:
        summary.append("<div style='margin: 30px 0 20px 0;'>")
        summary.append("<h3 style='color: #764ba2; border-bottom: 3px solid #764ba2; padding-bottom: 10px;'>💻 CPU-Only Resources (Sorted by Availability)</h3>")
        summary.append("</div>")
        
        summary.append("<div style='background: #f8f9fa; padding: 15px; border-radius: 10px;'>")
        
        for partition in cpu_partitions:
            dominant_state = 'idle' if partition['idle_nodes'] > partition['allocated_nodes'] else 'mix' if partition['idle_nodes'] > 0 else 'alloc'
            
            state_info = {
                'mix': {'emoji': '🟡', 'color': '#FFA726', 'text': 'Partially Available'},
                'alloc': {'emoji': '🔴', 'color': '#EF5350', 'text': 'Heavily Used'},
                'idle': {'emoji': '🟢', 'color': '#66BB6A', 'text': 'Available'},
                'down': {'emoji': '⚫', 'color': '#757575', 'text': 'Unavailable'}
            }.get(dominant_state, {'emoji': '⚪', 'color': '#BDBDBD', 'text': 'Unknown'})
            
            summary.append(
                f"<div style='padding: 15px; margin: 10px 0; background: white; border-left: 4px solid {state_info['color']}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
            )
            
            summary.append(
                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>"
                f"<div><strong style='font-size: 18px;'>{state_info['emoji']} {partition['name']}</strong> "
                f"<span style='color: {state_info['color']}; font-weight: 600; font-size: 14px;'>{state_info['text']}</span></div>"
                f"<div style='color: #666; font-size: 14px;'>⏱️ Time Limit: {partition['timelimit']}</div>"
                f"</div>"
            )
            
            # Node usage progress bar
            summary.append("<div style='margin: 10px 0;'>")
            summary.append("<div style='font-size: 13px; color: #666; margin-bottom: 5px; font-weight: 600;'>📦 Node Usage:</div>")
            summary.append(create_progress_bar(
                partition['allocated_nodes'],
                partition['total_nodes']
            ))
            summary.append("</div>")
            
            # CPU usage progress bar
            summary.append("<div style='margin: 10px 0;'>")
            summary.append("<div style='font-size: 13px; color: #666; margin-bottom: 5px; font-weight: 600;'>🔧 CPU Usage:</div>")
            summary.append(create_progress_bar(
                partition['allocated_cpus'],
                partition['total_cpus']
            ))
            summary.append("</div>")
            
            summary.append(
                f"<div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; font-size: 13px; color: #666;'>"
                f"<strong>Details:</strong> "
                f"Idle Nodes: {partition['idle_nodes']} | "
                f"Allocated Nodes: {partition['allocated_nodes']} | "
                f"Total CPUs: {partition['total_cpus']} | "
                f"Available CPUs: {partition['idle_cpus']}"
                f"</div>"
            )
            
            summary.append("</div>")
        
        summary.append("</div>")
    # If user requested GPU memory but nothing matched, show a warning


    # Filtered Results Summary Table
    if min_gpu_memory > 0 and filtered_results:
        # Sort by total available GPU memory (highest first)
        filtered_results.sort(key=lambda x: x['total_available_gpu_memory'], reverse=True)
        
        summary.append("<div style='margin-top: 30px; padding: 20px; background: #e8f5e9; border-radius: 10px; border: 2px solid #4CAF50;'>")
        summary.append(f"<h3 style='color: #2E7D32; margin-top: 0;'>✅ Best Available GPUs with ≥ {min_gpu_memory}GB Total Memory</h3>")
        summary.append("<p style='color: #666; margin-bottom: 15px;'>Sorted by total available GPU memory (highest first)</p>")
        summary.append("<table style='width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;'>")
        summary.append("<thead><tr style='background: #4CAF50; color: white;'>")
        summary.append("<th style='padding: 12px; text-align: left;'>Rank</th>")
        summary.append("<th style='padding: 12px; text-align: left;'>Partition</th>")
        summary.append("<th style='padding: 12px; text-align: center;'>Per GPU</th>")
        summary.append("<th style='padding: 12px; text-align: center;'>Total Available GPU Memory</th>")
        summary.append("<th style='padding: 12px; text-align: center;'>Available Nodes</th>")
        summary.append("<th style='padding: 12px; text-align: center;'>Idle Nodes</th>")
        summary.append("<th style='padding: 12px; text-align: left;'>Time Limit</th>")
        summary.append("</tr></thead><tbody>")
        
        for i, result in enumerate(filtered_results, 1):
            bg_color = "#f8f9fa" if i % 2 == 0 else "white"
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            summary.append(
                f"<tr style='background: {bg_color};'>"
                f"<td style='padding: 12px; border-bottom: 1px solid #ddd; font-size: 16px;'>{medal}</td>"
                f"<td style='padding: 12px; border-bottom: 1px solid #ddd;'><strong>{result['partition']}</strong></td>"
                f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #ddd;'>{result['gpu_memory']}GB</td>"
                f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #ddd;'><strong style='color: #4CAF50; font-size: 16px;'>{result['total_available_gpu_memory']}GB</strong><br><span style='font-size: 12px; color: #666;'>({result['gpu_memory']}GB × {result['available_nodes']} nodes)</span></td>"
                f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #ddd;'><span style='background: #2196F3; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600;'>{result['available_nodes']}</span></td>"
                f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #ddd;'><span style='background: #66BB6A; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600;'>{result['idle_nodes']}</span></td>"
                f"<td style='padding: 12px; border-bottom: 1px solid #ddd;'>{result['timelimit']}</td>"
                f"</tr>"
            )
        
        summary.append("</tbody></table>")
        summary.append("</div>")
    
    # State Legend
    summary.append("<div style='margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 10px;'>")
    summary.append("<h4 style='margin-top: 0; color: #333;'>📖 Legend</h4>")
    summary.append("<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;'>")
    
    legend_items = [
        ('🟢', '#66BB6A', 'Available', 'Resources ready to use'),
        ('🟡', '#FFA726', 'Partially Available', 'Some resources in use'),
        ('🔴', '#EF5350', 'Heavily Used', 'Most resources allocated'),
        ('⚫', '#757575', 'Unavailable', 'Offline or maintenance')
    ]
    
    for emoji, color, state, desc in legend_items:
        summary.append(
            f"<div style='background: white; padding: 10px; border-radius: 5px; border-left: 4px solid {color};'>"
            f"{emoji} <strong>{state}</strong>: {desc}"
            f"</div>"
        )
    
    summary.append("</div>")
    summary.append("</div>")
    
    return '\n'.join(summary)

def generate_slurm_script(job_name, account, partition, nodes, ntasks_per_node, 
                         cpus_per_task, memory, walltime, program_file, program_args,
                         gpu_count, output_file, error_file, combine_output,
                         array_indices, dependency_type, dependency_job_ids,
                         mail_type, mail_user, export_env, nodelist, signal_time):
    
    # Enhanced error checking for required fields
    errors = []
    warnings = []
    
    # Required field validation
    if not job_name or not job_name.strip():
        errors.append("Job Name is required")
    elif len(job_name.strip()) > 64:
        warnings.append("Job Name is longer than 64 characters (may be truncated)")
    
    if not program_file or not program_file.strip():
        errors.append("Program/Script to Run is required")
    elif not (program_file.strip().endswith(('.py', '.sh', '.R', '.m', '.cpp', '.c', '.f90', '.f', '.pl', '.rb', '.go', '.rs')) or 
              program_file.strip().startswith(('./', '/'))):
        warnings.append("Program file doesn't have a common extension - ensure it's executable")
    
    if not walltime:
        errors.append("Wall Time is required")
    else:
        # Validate walltime format
        try:
            parts = walltime.split(':')
            if len(parts) != 3:
                errors.append("Wall Time must be in HH:MM:SS format")
            else:
                hours, minutes, seconds = map(int, parts)
                if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                    errors.append("Invalid time values in Wall Time")
        except ValueError:
            errors.append("Wall Time must contain only numbers and colons (HH:MM:SS)")
    
    # Additional validations with proper error handling
    try:
        nodes_int = int(nodes)
        if nodes_int < 1:
            errors.append("Number of nodes must be at least 1")
    except (ValueError, TypeError):
        errors.append("Invalid nodes value")
    
    try:
        gpu_count_int = int(gpu_count)
        if gpu_count_int > 0 and (not partition or partition == "Default"):
            warnings.append("GPU requested but no GPU partition selected")
    except (ValueError, TypeError):
        errors.append("Invalid GPU count value")
    
    if array_indices and array_indices.strip():
        try:
            if not any(c.isdigit() for c in array_indices):
                errors.append("Array indices must contain numbers")
        except:
            errors.append("Invalid array indices format")
    
    if dependency_type != "None" and (not dependency_job_ids or not dependency_job_ids.strip()):
        errors.append("Dependency Job IDs required when dependency type is selected")
    
    if mail_type != "None" and (not mail_user or not mail_user.strip()):
        warnings.append("Email address recommended when email notifications are enabled")
    
    # Return errors or warnings
    if errors:
        error_msg = "❌ ERRORS FOUND:\n" + "\n".join(f"• {error}" for error in errors)
        if warnings:
            error_msg += "\n\n⚠️ WARNINGS:\n" + "\n".join(f"• {warning}" for warning in warnings)
        return error_msg
    
    if warnings:
        warning_msg = "⚠️ WARNINGS:\n" + "\n".join(f"• {warning}" for warning in warnings) + "\n\n"
    else:
        warning_msg = ""
    
    # Start building the script
    script = "#!/bin/bash\n\n"
    script += "# Slurm job script generated by GUI\n"
    
    if warnings:
        for line in warning_msg.strip().split('\n'):
            if line:
                script += f"# {line}\n"
        script += "#\n"
    
    script += "\n"
    
    # Required directives
    script += f"#SBATCH --job-name={job_name}\n"
    script += f"#SBATCH --time={walltime}\n"
    
    # Optional account
    if account and account.strip():
        script += f"#SBATCH --account={account}\n"
    
    # Partition/Queue
    if partition and partition != "Default":
        script += f"#SBATCH --partition={partition}\n"
    
    # Resource allocation with proper error handling
    try:
        nodes_int = int(nodes)
        ntasks_int = int(ntasks_per_node)
        cpus_int = int(cpus_per_task)
        gpu_int = int(gpu_count)
        signal_int = int(signal_time)
        
        script += f"#SBATCH --nodes={nodes_int}\n"
        if ntasks_int > 0:
            script += f"#SBATCH --ntasks-per-node={ntasks_int}\n"
        if cpus_int > 0:
            script += f"#SBATCH --cpus-per-task={cpus_int}\n"
        
        # Memory
        if memory and memory != "Default":
            script += f"#SBATCH --mem={memory}\n"
        
        # GPU resources
        if gpu_int > 0:
            script += f"#SBATCH --gres=gpu:{gpu_int}\n"
    except (ValueError, TypeError):
        return "❌ ERROR: Invalid numeric values in resource allocation"
    
    # Output/Error files
    if combine_output:
        output_name = output_file if output_file.strip() else f"{job_name}_%j.out"
        script += f"#SBATCH --output={output_name}\n"
    else:
        output_name = output_file if output_file.strip() else f"{job_name}_%j.out"
        error_name = error_file if error_file.strip() else f"{job_name}_%j.err"
        script += f"#SBATCH --output={output_name}\n"
        script += f"#SBATCH --error={error_name}\n"
    
    # Job arrays
    if array_indices and array_indices.strip():
        script += f"#SBATCH --array={array_indices}\n"
    
    # Job dependencies
    if dependency_type != "None" and dependency_job_ids and dependency_job_ids.strip():
        script += f"#SBATCH --dependency={dependency_type}:{dependency_job_ids}\n"
    
    # Email notifications
    if mail_type != "None":
        script += f"#SBATCH --mail-type={mail_type}\n"
        if mail_user and mail_user.strip():
            script += f"#SBATCH --mail-user={mail_user}\n"
    
    # Environment export
    if export_env != "Default":
        script += f"#SBATCH --export={export_env}\n"
    
    # Specific node list
    if nodelist and nodelist.strip():
        script += f"#SBATCH --nodelist={nodelist}\n"
    
    # Signal before job termination
    if signal_int > 0:
        script += f"#SBATCH --signal=B:USR1@{signal_int}\n"
    
    script += "\n"
    
    # Add environment variable examples as comments
    script += "# Available Slurm environment variables:\n"
    script += "# $SLURM_JOB_NAME - Job name\n"
    script += "# $SLURM_JOB_ID - Job ID\n"
    script += "# $SLURM_SUBMIT_DIR - Submit directory\n"
    script += "# $SLURM_SUBMIT_HOST - Submit host\n"
    script += "# $SLURM_JOB_NODELIST - Node list\n"
    script += "# $SLURM_JOB_PARTITION - Partition name\n"
    script += "# $SLURM_JOB_NUM_NODES - Number of allocated nodes\n"
    script += "# $SLURM_NTASKS - Number of processes\n"
    script += "# $SLURM_TASKS_PER_NODE - Processes per node\n"
    script += "# $SLURM_ARRAY_TASK_ID - Array task ID (if array job)\n\n"
    
    # Change to submit directory
    script += "# Change to the directory from which the job was submitted\n"
    script += "cd $SLURM_SUBMIT_DIR\n\n"
    
    # Load modules section
    script += "# Load required modules here\n"
    script += "# module load python/3.9\n"
    script += "# module load gcc/9.3.0\n\n"
    
    # Main program execution
    script += "# Run the program\n"
    if program_file.startswith('./') or program_file.startswith('/'):
        script += f"python {program_file}"
    else:
        script += f"python ./{program_file}"
    
    if program_args and program_args.strip():
        script += f" {program_args}"
    
    script += "\n"
    
    return script

# Define all the input components
def create_interface():
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    .gr-box {
        border-radius: 8px !important;
    }
    h1, h2, h3, h4 {
        font-weight: 600 !important;
    }
    """
    
    with gr.Blocks(title="🚀 Slurm Script Generator", theme=gr.themes.Soft(), css=custom_css) as interface:
        
        # Main header
        gr.HTML("""
        <div style='text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);'>
            <h1 style='margin: 0; font-size: 42px; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>🚀 Slurm Script Generator</h1>
            <p style='margin: 15px 0 0 0; font-size: 18px; opacity: 0.95;'>Generate professional Slurm job scripts and monitor cluster resources</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # Tab 1: Script Generator
            with gr.TabItem("📝 Script Generator", id=0):
                gr.Markdown("""
                ### Quick Start Guide
                
                1. **Fill Required Fields** (marked with ⭐)
                2. **Configure Resources** based on your job needs
                3. **Generate Script** and copy the output
                4. **Submit** using: `sbatch your_script.sh`
                
                💡 **Tip:** Check the Resource Monitor tab to find available GPUs before submitting!
                """)
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Basic Information")
                        job_name = gr.Textbox(
                            label="⭐ Job Name", 
                            placeholder="my_training_job", 
                            info="Required: A descriptive name for your job"
                        )
                        account = gr.Textbox(
                            label="Account", 
                            placeholder="your_account_name",
                            info="Optional: Billing account"
                        )
                        partition = gr.Dropdown(
                            choices=["Default", "standard", "gpuq", "a5000", "a5500", "a100", "h200", "v100", "2080ti", 
                                   "a5000_w", "a5500_w", "l40s_nova", "l40s_indrani", "preemptable"],
                            value="Default",
                            label="Partition/Queue",
                            info="Select compute partition (check Resource Monitor)"
                        )
                        program_file = gr.Textbox(
                            label="⭐ Program/Script", 
                            placeholder="train.py",
                            info="Required: Script or executable to run"
                        )
                        program_args = gr.Textbox(
                            label="Program Arguments", 
                            placeholder="--config config.yaml --epochs 100",
                            info="Optional: Command line arguments"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Compute Resources")
                        nodes = gr.Slider(
                            minimum=1, maximum=20, value=1, step=1, 
                            label="🖥️ Number of Nodes"
                        )
                        ntasks_per_node = gr.Slider(
                            minimum=0, maximum=128, value=1, step=1, 
                            label="Tasks per Node",
                            info="0 = auto (typically 1 for GPU jobs)"
                        )
                        cpus_per_task = gr.Slider(
                            minimum=0, maximum=64, value=1, step=1, 
                            label="CPUs per Task",
                            info="0 = auto"
                        )
                        memory = gr.Dropdown(
                            choices=["Default", "1G", "2G", "4G", "8G", "16G", "32G", "64G", "128G", "256G"],
                            value="8G",
                            label="💾 Memory per Node"
                        )
                        walltime = gr.Dropdown(
                            choices=["00:15:00", "00:30:00", "01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00", "24:00:00"],
                            value="01:00:00",
                            label="⭐ ⏱️ Wall Time (HH:MM:SS)",
                            info="Required: Maximum runtime",
                            allow_custom_value=True
                        )
                        gpu_count = gr.Slider(
                            minimum=0, maximum=8, value=0, step=1, 
                            label="🎮 Number of GPUs",
                            info="0 = no GPU"
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Output Configuration")
                        output_file = gr.Textbox(
                            label="Output File", 
                            placeholder="output_%j.out",
                            info="%j = job ID"
                        )
                        error_file = gr.Textbox(
                            label="Error File", 
                            placeholder="error_%j.err"
                        )
                        combine_output = gr.Checkbox(
                            label="Combine stdout and stderr into one file", 
                            value=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔗 Advanced: Arrays & Dependencies")
                        array_indices = gr.Textbox(
                            label="Array Job Indices", 
                            placeholder="1-10 or 1,3,5-8",
                            info="For parameter sweeps"
                        )
                        dependency_type = gr.Dropdown(
                            choices=["None", "after", "afterok", "afternotok", "afterany"],
                            value="None",
                            label="Job Dependency Type"
                        )
                        dependency_job_ids = gr.Textbox(
                            label="Dependency Job IDs", 
                            placeholder="12345,12346"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📧 Email Notifications")
                        mail_type = gr.Dropdown(
                            choices=["None", "BEGIN", "END", "FAIL", "ALL", "BEGIN,END", "END,FAIL", "BEGIN,END,FAIL"],
                            value="FAIL",
                            label="Notification Events"
                        )
                        mail_user = gr.Textbox(
                            label="Email Address", 
                            placeholder="user@university.edu"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 Advanced Options")
                        export_env = gr.Dropdown(
                            choices=["Default", "ALL", "NONE"],
                            value="Default",
                            label="Export Environment Variables"
                        )
                        nodelist = gr.Textbox(
                            label="Specific Nodes", 
                            placeholder="node001,node002",
                            info="Force specific nodes (usually not needed)"
                        )
                        signal_time = gr.Slider(
                            minimum=0, maximum=300, value=0, step=10, 
                            label="Signal Before Timeout (seconds)",
                            info="0 = disabled"
                        )
                
                # Submit button
                submit_btn = gr.Button("🚀 Generate Slurm Script", variant="primary", size="lg", scale=2)
                
                # Output
                gr.Markdown("### 📄 Generated Script")
                output = gr.Textbox(
                    label="Copy this script to a .sh file", 
                    lines=25, 
                    max_lines=35,
                    show_copy_button=True
                )
                
                # Connect the function
                submit_btn.click(
                    fn=generate_slurm_script,
                    inputs=[
                        job_name, account, partition, nodes, ntasks_per_node, cpus_per_task,
                        memory, walltime, program_file, program_args, gpu_count, output_file,
                        error_file, combine_output, array_indices, dependency_type,
                        dependency_job_ids, mail_type, mail_user, export_env, nodelist, signal_time
                    ],
                    outputs=output
                )
            
            # Tab 2: Resource Monitor
            with gr.TabItem("📊 Resource Monitor", id=1):
                gr.Markdown("""
                ### 🎮 Real-Time Cluster Resource Monitor
                
                View detailed resource usage with progress bars showing exact node and CPU allocation.
                
                **Filter Logic:** Total Available GPU Memory = Per-GPU Memory × Available Nodes
                - Example: 24GB GPU with 2 available nodes = 48GB total available memory
                - Partitions are ranked by total available GPU memory (highest first)
                """)
                
                with gr.Row():
                    gpu_memory_filter = gr.Slider(
                        minimum=0, maximum=150, value=0, step=1,
                        label="🔍 Filter by Minimum Total Available GPU Memory (GB)",
                        info="Total Available = Per-GPU Memory × Available Nodes | Set to 0 to show all",
                        scale=3
                    )
                    refresh_btn = gr.Button("🔄 Refresh Resources", variant="primary", size="lg", scale=1)
                
                resource_output = gr.HTML(
                    value="<p style='text-align: center; padding: 40px; color: #666;'>Click '🔄 Refresh Resources' to load current cluster status...</p>"
                )
                
                # Auto-refresh function
                def refresh_resources(min_gpu_mem):
                    return get_available_resources(min_gpu_mem)
                
                refresh_btn.click(
                    fn=refresh_resources,
                    inputs=[gpu_memory_filter],
                    outputs=resource_output
                )
                
                gpu_memory_filter.change(
                    fn=refresh_resources,
                    inputs=[gpu_memory_filter],
                    outputs=resource_output
                )
                
                gr.Markdown("---")
                
                # Quick reference in columns
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 💾 Common GPU Memory Sizes
                        
                        | GPU Model | Memory |
                        |-----------|--------|
                        | RTX 2080 Ti | 11 GB |
                        | V100 | 16 GB |
                        | A5000 | 24 GB |
                        | A5500 | 24 GB |
                        | A100 | 40 GB |
                        | L40S | 48 GB |
                        | H200 | 141 GB |
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 💡 How Filtering Works
                        
                        **Total Available GPU Memory = Per-GPU Memory × Available Nodes**
                        
                        **Examples:**
                        - A5000 (24GB) with 2 nodes = **48GB total** ✅
                        - V100 (16GB) with 3 nodes = **48GB total** ✅
                        - 2080Ti (11GB) with 1 node = **11GB total** ❌
                        
                        **When you set filter to 40GB:**
                        - Shows partitions with ≥40GB total available
                        - Ranked by total GPU memory (highest first)
                        - Considers both per-GPU size and node count
                        """)
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; padding: 20px; margin-top: 30px; color: #666; border-top: 1px solid #ddd;'>
            <p style='margin: 0;'>💻 Made with Gradio | 📚 <a href='https://slurm.schedmd.com/' target='_blank'>Slurm Documentation</a></p>
        </div>
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
