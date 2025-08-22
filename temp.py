import os
import json
import math
from typing import List, Dict, Any
from rectpack import newPacker, SORT_NONE
from rectpack.skyline import SkylineBl

def pack_to_min_area(rectangles, allow_rotation=False):
    """
    Given a list of (w,h) rectangles, finds the arrangement
    that minimizes the bounding‐box area by scanning possible bin widths.

    Returns: (best_area, best_width, best_height, placements)
      where placements is a list of (rid, w, h, x, y).
    """
    # 1. Compute bounds on the bin width & height
    min_width = max(w for w,h in rectangles)
    max_width = sum(w for w,h in rectangles)
    # if you want, you could also scan heights; here we fix height to the sum of heights
    max_height = sum(h for w,h in rectangles)

    best = (float("inf"), None)  # (area, data)

    # 2. Try every candidate width
    for W in range(min_width, max_width + 1):
        packer = newPacker(rotation=allow_rotation)
        # give one bin of width W and a “tall” height
        packer.add_bin(W, max_height, count=1)
        for rid, (w, h) in enumerate(rectangles):
            packer.add_rect(w, h, rid)
        packer.pack()

        abin = packer[0]
        if len(abin) < len(rectangles):
            # at this width W, not all rectangles fit → skip
            continue

        # compute the true bounding box of the placed rectangles
        max_x = max(rect.x + rect.width  for rect in abin)
        max_y = max(rect.y + rect.height for rect in abin)
        area  = max_x * max_y

        if area < best[0]:
            # record the arrangement
            placements = [(r.rid, r.width, r.height, r.x, r.y) for r in abin]
            best = (area, (W, max_x, max_y, placements))

    if best[1] is None:
        raise ValueError("No packing found")
    _, (W, used_w, used_h, placements) = best
    return best[0], used_w, used_h, placements


def compute_pack(rectangles):
    # 2. Compute a safe “big bin” in floats
    max_bin_width  = sum(w for w, l in rectangles)
    max_bin_length = sum(l for w, l in rectangles)

    # 3. Create the packer & add a single float‐sized bin
    packer = newPacker(rotation=False)
    packer.add_bin(max_bin_width, max_bin_length, count=1)

    # 4. Add each rectangle—no ceil, just floats
    for rid, (w, l) in enumerate(rectangles):
        packer.add_rect(w, l, rid)

    # 5. Pack
    packer.pack()

    # 6. Extract placements & compute true bounding box
    placements = []
    used_w = used_l = 0.0
    for abin in packer:
        for rect in abin:
            x, y = rect.x, rect.y            # floats
            w, l = rect.width, rect.height  # floats
            placements.append((rect.rid, w, l, x, y))
            used_w = max(used_w, x + w)
            used_l = max(used_l, y + l)
    
    return used_w, used_l, placements

# # Example usage:
# rects = [(100,50), (80,30), (60,40), (30,60)]
# area, w, h, placements = pack_to_min_area(rects, allow_rotation=False)


def skyline_pack_algorithm(machines,margin):
    
    # 1. Inflate every footprint by 2*margin
    inflated = [(w + 2*margin, l + 2*margin, rid)
                for rid, (w, l) in enumerate(machines)]
    
    # 2. Create a packer with Skyline bottom-left & no sort 
    packer = newPacker(pack_algo=SkylineBl, sort_algo=SORT_NONE, rotation=False)

    # 3. Add a single large bin
    maxW = sum(w for w,l,_ in inflated)
    maxL = sum(l for w,l,_ in inflated)
    packer.add_bin(maxW, maxL, count=1)

    # 4. Add inflated rectangles in sequence
    for w_inf, l_inf, rid in inflated:
        packer.add_rect(w_inf, l_inf, rid)

    # 5. Perform the packing
    packer.pack()

    # 6. Get exact placements of the real machines back.
    placements = []
    for abin in packer:
        for r in abin:
            rid = r.rid
            x_inf, y_inf = r.x, r.y
            #insert machine placements by margin
            placements.append((
                rid,
                machines[rid][0], machines[rid][1],
                x_inf + margin, y_inf + margin
            ))

    # 7. Computer minimal enclosing box
    width_min_box  = max(x + w for _, w, l, x, y in placements)
    length_min_box = max(y + l for _, w, l, x, y in placements)

    # 8. Return the width and the length of the cluster, also the placements of the machines.
    return width_min_box, length_min_box, placements



def convert_substeps_and_machines(workflow: Dict, machines: List[Dict]) -> Dict[str, Any]:
    """
    Convert workflow and machines data into the factory plan format with machine clusters.
    """
    # Convert workflow steps to factory plan steps
    steps = []
    workflow_steps = workflow.get("steps", []) if isinstance(workflow, dict) else []
    
    for step in workflow_steps:
        try:
            step_number = int(step.get("step_number", 0))
            
            # Get suggested machines from workflow step
            suggested_machines = step.get("suggested_machines", [])
            if not isinstance(suggested_machines, list):
                suggested_machines = []
            
            # Create factory plan step
            factory_step = {
                "step_number": step_number,
                "operation_name": step.get("step_name", ""),
                "operation_type": step.get("step_type", "manual").lower(),
                "description": step.get("step_description", ""),
                "quality_control_measures": step.get("process_understanding", "Quality checks to be determined"),
                "suggested_machines": [m for m in suggested_machines if m]  # Filter empty strings
            }
            steps.append(factory_step)
        except (ValueError, TypeError):
            continue
    
    # Group machines by step number
    step_machines = {}
    total_machines_processed = 0
    
    print(f"Processing {len(machines)} machine items...")
    print(f"Processing flattened machine structure with {len(machines)} machines")
    for machine in machines:
        if not isinstance(machine, dict):
            continue
            
        matched_machine = machine.get("matched_machine", {})
        if matched_machine and isinstance(matched_machine, dict):
            machine_data = matched_machine
        elif machine.get("estimated_machine") is not None:
            machine_data = machine
        else:
            continue
        if not machine_data.get("estimated_machine", False):
            continue
            
        try:
            step_number_str = machine.get("step_number", "0")
            if isinstance(step_number_str, str) and "." in step_number_str:
                step_number = int(float(step_number_str))
            else:
                step_number = int(float(step_number_str))
            
            if step_number not in step_machines:
                step_machines[step_number] = []
            
            step_machines[step_number].append(machine_data)
            total_machines_processed += 1
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse step number '{step_number_str}': {e}")
            continue
    
    # Create clustered machines
    machine_list = []
    total_cost = 0.0
    power_requirements = []
    
    for step_number, machines_in_step in step_machines.items():
        if not machines_in_step:
            continue
            
        # Initialize cluster data with first machine
        rectangles = []
        first_machine = machines_in_step[0]
        cluster = {
            "name": [first_machine.get("machine_name", "")],
            "primary_function": [first_machine.get("primary_function", "")],
            "estimated_cost": float(first_machine.get("estimated_cost", 0)),
            "make_model": [first_machine.get("manufacturer", "Generic Model")],
            "electricity_requirement": float(first_machine.get("power", 0)),
            "estimated_size": {
                "length": None,
                "width": None,
                "height": float(first_machine.get("size_height", 0))
            },
            "citation_link": ["N/A"],
            "information_score": 0.8889,
            "interior_placements": None
        }
        # if math.ceil(float(first_machine.get("size_width", 0))) > 0 and math.ceil(float(first_machine.get("size_length", 0)))>0:
        #     rectangles.append((math.ceil(float(first_machine.get("size_width", 0))),math.ceil(float(first_machine.get("size_length", 0)))))
        rectangles.append((float(first_machine.get("size_width")),float(first_machine.get("size_length"))))
        # Add other machines to the cluster
        for machine in machines_in_step[1:]:
            cluster["name"].append(machine.get("machine_name", ""))
            cluster["primary_function"].append(machine.get("primary_function", ""))
            cluster["estimated_cost"] += float(machine.get("estimated_cost", 0))
            cluster["make_model"].append(machine.get("manufacturer", "Generic Model"))
            cluster["electricity_requirement"] += float(machine.get("power", 0))
            
            # Add size components
            # if math.ceil(machine.get("size_width", 0)) > 0 and math.ceil(machine.get("size_length", 0))>0:
            #     rectangles.append((math.ceil(machine.get("size_width", 0)),math.ceil(machine.get("size_length", 0))))
            rectangles.append((float(machine.get("size_width")),float(machine.get("size_length"))))
            cluster["estimated_size"]["height"] += float(machine.get("size_height", 0))  #TODO : Take care of height packing, when we move to 3D. OK for now!!
            
            cluster["citation_link"].append("N/A")
        # print(rectangles)
        # area, w, l, placements = pack_to_min_area(rectangles,False)
        w, l, placements = skyline_pack_algorithm(rectangles, margin=0.75) # Change margin as per you need 
        # print(rectangles , "Pack Area --> " , w, "X", l)
        
        #Assign the packed area to the cluster as the new size of the machines.
        cluster["estimated_size"]["length"] = l
        cluster["estimated_size"]["width"]  = w
        # cluster["machine_placements"] = placements
        # Format the cluster for output
        formatted_cluster = {
            "name": ", ".join(cluster["name"]),
            "primary_function": ", ".join(cluster["primary_function"]),
            "estimated_cost": cluster["estimated_cost"],
            "make_model": ", ".join(cluster["make_model"]),
            "electricity_requirement": f"{cluster['electricity_requirement']} kW",
            "estimated_size": f"{cluster['estimated_size']['length']} x {cluster['estimated_size']['width']} x {cluster['estimated_size']['height']}",
            "workflow_step": step_number,
            "citation_link": ", ".join(cluster["citation_link"]),
            "option": 1,
            "information_score": cluster["information_score"],
            "interior_placements": placements
        }
        
        machine_list.append(formatted_cluster)
        total_cost += cluster["estimated_cost"]
        power_requirements.append(cluster["electricity_requirement"])
    
    # Calculate total power requirements
    total_power = sum(power_requirements)
    power_range = f"{total_power * 0.8}-{total_power * 1.2} kW Estimated" if power_requirements else "0 kW"
    
    result = {
        "factory_plan": {
            "steps": sorted(steps, key=lambda x: x["step_number"]),
            "total_steps": len(steps),
            "estimated_completion_time": "1 work shift (8 hours)"
        },
        "machine_list": {
            "machines": machine_list,
            "original_machine_count": len(machines),
            "total_cost": total_cost,
            "power_requirements": power_range
        }
    }
    
    # Save the result to a JSON file in the current directory
    local_folder = 'local_data'
    os.makedirs(local_folder, exist_ok=True)
    output_file = "factory_plan_clustered.json"
    out_path = os.path.join(local_folder,output_file)
    with open(out_path,'w') as f:
        json.dump(result,f, indent=2)
    
    print(f"Successfully saved clustered factory plan to {output_file}")

    return result

def main():
    # Load input data from JSON file
    input_file = "input_data.json"
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract substeps and machines from input data
        workflow = data.get("workflow", [])
        machines = data.get("machines", [])
        
        # Convert the data (which will automatically save the output)
        convert_substeps_and_machines(workflow, machines)
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

