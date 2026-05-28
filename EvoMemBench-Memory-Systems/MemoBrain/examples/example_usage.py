"""
Basic Usage Example for MemoBrain

This example demonstrates how to:
1. Load an existing memory snapshot
2. Visualize the memory structure
3. Inspect nodes and edges in the memory graph
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memobrain import MemoBrain


def main():
    print("="*100)
    print("MemoBrain Basic Usage Example")
    print("="*100)
    print()
    
    # Step 1: Initialize MemoBrain
    # Note: You need to have a MemoBrain model running on localhost:8002
    print("Step 1: Initializing MemoBrain...")
    memory = MemoBrain(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1",
        model_name="MemoBrain-8B"
    )
    print("✓ MemoBrain initialized\n")
    
    # Step 2: Load existing memory graph from snapshot
    print("Step 2: Loading memory snapshot...")
    snapshot_path = os.path.join(os.path.dirname(__file__), "memory_snapshot.json")
    
    if not os.path.exists(snapshot_path):
        print(f"Error: Memory snapshot not found at {snapshot_path}")
        return
    
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    
    memory.load_dict_memory(data["memory"])
    print(f"✓ Loaded memory with {len(memory.graph.nodes)} nodes and {len(memory.graph.edges)} edges\n")
    
    # Step 3: Visualize memory structure
    print("Step 3: Memory Graph Structure")
    print("="*100)
    print(memory.graph.pretty_print())
    print("="*100)
    print()
    
    # Step 4: Inspect individual nodes
    print("Step 4: Inspecting Individual Nodes")
    print("-"*100)
    
    # Get all node IDs
    node_ids = list(memory.graph.nodes.keys())
    
    # Show first few nodes in detail
    for node_id in node_ids[:3]:  # Show first 3 nodes
        node = memory.graph.nodes[node_id]
        print(f"\nNode {node_id}:")
        print(f"  Kind: {node.kind}")
        print(f"  Active: {node.active}")
        print(f"  Related Turn IDs: {node.related_turn_ids}")
        
        if node.kind == "task":
            print(f"  Thought: {node.thought[:100]}..." if len(node.thought) > 100 else f"  Thought: {node.thought}")
        else:
            print(f"  Thought Messages: {len(node.thought)} messages")
            if node.thought:
                # Show first message
                first_msg = node.thought[0]
                content_preview = first_msg['content'][:80] + "..." if len(first_msg['content']) > 80 else first_msg['content']
                print(f"    - {first_msg['role']}: {content_preview}")
    
    print("\n" + "-"*100)
    print()
    
    # Step 5: Inspect edges (dependencies)
    print("Step 5: Memory Dependencies (Edges)")
    print("-"*100)
    
    for i, edge in enumerate(memory.graph.edges[:5]):  # Show first 5 edges
        print(f"\nEdge {i+1}:")
        print(f"  From Node {edge.src} → To Node {edge.dst}")
        print(f"  Rationale: {edge.rationale[:80]}..." if len(edge.rationale) > 80 else f"  Rationale: {edge.rationale}")
    
    print("\n" + "-"*100)
    print()
    
    # Step 6: Statistics
    print("Step 6: Memory Statistics")
    print("-"*100)
    
    # Count nodes by kind
    node_kinds = {}
    for node in memory.graph.nodes.values():
        node_kinds[node.kind] = node_kinds.get(node.kind, 0) + 1
    
    print(f"Total Nodes: {len(memory.graph.nodes)}")
    print(f"Total Edges: {len(memory.graph.edges)}")
    print(f"\nNodes by Kind:")
    for kind, count in node_kinds.items():
        print(f"  - {kind}: {count}")
    
    # Count active vs inactive nodes
    active_nodes = sum(1 for node in memory.graph.nodes.values() if node.active)
    print(f"\nActive Nodes: {active_nodes}")
    print(f"Inactive Nodes: {len(memory.graph.nodes) - active_nodes}")
    
    print("\n" + "="*100)
    print()
    
    # Step 7: Export memory back to dict
    print("Step 7: Exporting Memory")
    print("-"*100)
    
    exported_memory = memory.graph.to_dict()
    print(f"✓ Memory exported successfully")
    print(f"  - Nodes: {len(exported_memory['nodes'])}")
    print(f"  - Edges: {len(exported_memory['edges'])}")
    
    # Optionally save to a new file
    # output_path = os.path.join(os.path.dirname(__file__), "memory_export_example.json")
    # with open(output_path, 'w') as f:
    #     json.dump({"memory": exported_memory}, f, indent=2)
    # print(f"✓ Saved exported memory to: {output_path}")
    
    # print("\n" + "="*100)
    # print("\n✨ Example completed successfully!\n")


if __name__ == "__main__":
    main()
