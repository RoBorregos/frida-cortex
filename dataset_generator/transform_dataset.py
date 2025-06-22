"""
Semi automatic transformation of the dataset to the new format, doesn't
cover ground thruth of retrieval of items and locations.
It is needed to previously create the dataset_execution_auto.json file from
dataset.json
"""
import json
import os
import random

def get_item_name(item_to_pick, items_data):
    # Simplified logic to get an item name.
    # In a real scenario, this would involve embeddings.
    # For now, let's just try to find a close match or return the query.
    for item in items_data:
        if item_to_pick.lower() in item["document"].lower():
            return item["document"]
    return item_to_pick

def get_location_string(location, areas_data):
    # Simplified logic to get a location string.
    # For now, just return the location. A more complex implementation could
    # try to match and format it as area -> subarea.
    return location

def transform_command(cmd, string_cmd, items_data, areas_data):
    if "success" in cmd and "result" in cmd:
        return cmd

    action = cmd.get("action")
    new_cmd = {"action": action, "success": True}

    if action == "go_to":
        location = cmd.get("location_to_go")
        if location:
            result = f"arrived to: {get_location_string(location, areas_data)}"
        else:
            result = "failed to arrive to place"
            new_cmd["success"] = False
        new_cmd["result"] = result

    elif action == "pick_object":
        obj = cmd.get("object_to_pick")
        if obj:
            item_name = get_item_name(obj, items_data)
            result = f"picked up: {item_name}"
        else:
            result = "failed to pick up object"
            new_cmd["success"] = False
        new_cmd["result"] = result

    elif action == "place_object":
        new_cmd["result"] = "placed object"

    elif action == "say_with_context":
        info = cmd.get("previous_command_info")
        if info == "get_person_info":
             # This is a bit of a guess from the examples.
            new_cmd["result"] = "person name is John"
        elif info == "count":
            # Another guess from examples.
            new_cmd["result"] = "there are 4 items"
        elif info == "get_visual_info":
            new_cmd["result"] = "found: biggest object" # placeholder
        elif info:
             new_cmd["result"] = f"{info}"
        else:
            new_cmd["result"] = "said something"

    elif action == "answer_question":
        new_cmd["result"] = "answered user's question"

    elif action == "get_visual_info":
        measure = cmd.get("measure")
        obj_cat = cmd.get("object_category")
        new_cmd["result"] = f"found: {measure} {obj_cat}"

    elif action == "give_object":
        new_cmd["result"] = "object given"

    elif action == "follow_person_until":
        dest = cmd.get("destination")
        if dest == "canceled" or dest == "cancelled":
            new_cmd["result"] = "followed user until cancelled"
        else:
            new_cmd["result"] = f"arrived to: {get_location_string(dest, areas_data)}"

    elif action == "guide_person_to":
        dest = cmd.get("destination_room")
        new_cmd["result"] = f"arrived to: {get_location_string(dest, areas_data)}"

    elif action == "get_person_info":
        info_type = cmd.get("info_type")
        if info_type == "gesture":
            new_cmd["result"] = "person gesture is pointing to the right"
        elif info_type == "pose":
            new_cmd["result"] = "person pose is standing"
        elif info_type == "name":
            new_cmd["result"] = "person name is John"
        else:
            new_cmd["result"] = f"person {info_type} was found"

    elif action == "count":
        target = cmd.get("target_to_count")
        # The task implementation always returns 4.
        new_cmd["result"] = f"found: 4 {target}"

    elif action == "find_person":
        attr = cmd.get("attribute_value")
        if attr and attr != "":
            new_cmd["result"] = f"found person with attribute: {attr}"
        else:
            new_cmd["result"] = "found person"

    elif action == "find_person_by_name":
        name = cmd.get("name")
        new_cmd["result"] = f"found {name}"

    else:
        # If action is unknown, mark as failed
        new_cmd["success"] = False
        new_cmd["result"] = f"unknown action: {action}"

    return new_cmd


def main():
    # Construct paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir) # Assumes script is in dataset_generator

    dataset_path = os.path.join(script_dir, "dataset_execution_auto.json")
    items_path = os.path.join(workspace_root, "command_interpreter", "embeddings", "dataframes", "items.json")
    areas_path = os.path.join(workspace_root, "command_interpreter", "embeddings", "maps", "areas.json")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    with open(items_path, 'r') as f:
        items_data = json.load(f)
    
    with open(areas_path, 'r') as f:
        areas_data = json.load(f)


    for entry in data:
        if "structured_cmd" in entry:
            new_structured_cmd = []
            # Check if already transformed
            if entry["structured_cmd"] and "success" in entry["structured_cmd"][0]:
                continue
            for cmd in entry["structured_cmd"]:
                transformed = transform_command(cmd, entry["string_cmd"], items_data, areas_data)
                new_structured_cmd.append(transformed)
            entry["structured_cmd"] = new_structured_cmd

    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print("Transformation complete. `dataset_execution_auto.json` has been updated.")


if __name__ == "__main__":
    main() 