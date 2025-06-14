import random
import re
import itertools
import warnings

class CommandGenerator:

    def __init__(self, person_names, location_names, placement_location_names, room_names, object_names,
                 object_categories_plural, object_categories_singular):
        self.person_names = person_names
        self.location_names = location_names
        self.placement_location_names = placement_location_names
        self.room_names = room_names
        self.object_names = object_names
        self.object_categories_plural = object_categories_plural
        self.object_categories_singular = object_categories_singular

    # ------------------------------
    # Dictionaries and lists (unchanged)
    # ------------------------------
    verb_dict = {
        "take": ["take", "get", "grasp", "fetch"],
        "place": ["put", "place"],
        "deliver": ["bring", "give", "deliver"],
        "bring": ["bring", "give"],
        "go": ["go", "navigate"],
        "find": ["find", "locate", "look for"],
        "talk": ["tell", "say"],
        "answer": ["answer"],
        "meet": ["meet"],
        "tell": ["tell"],
        "greet": ["greet", "salute", "say hello to", "introduce yourself to"],
        "remember": ["meet", "contact", "get to know", "get acquainted with"],
        "count": ["tell me how many"],
        "describe": ["tell me how", "describe"],
        "offer": ["offer"],
        "follow": ["follow"],
        "guide": ["guide", "escort", "take", "lead"],
        "accompany": ["accompany"]
    }
    
    all_cmd_types = [
        "goToLoc", "takeObjFromPlcmt", "findPrsInRoom", "findObjInRoom",
        "meetPrsAtBeac", "countObjOnPlcmt", "countPrsInRoom", "tellPrsInfoInLoc", "tellObjPropOnPlcmt", "talkInfoToGestPrsInRoom",
        "answerToGestPrsInRoom", "followNameFromBeacToRoom", "guideNameFromBeacToBeac",
        "guidePrsFromBeacToBeac", "guideClothPrsFromBeacToBeac", "bringMeObjFromPlcmt",
        "tellCatPropOnPlcmt", "greetClothDscInRm", "greetNameInRm", "meetNameAtLocThenFindInRm",
        "countClothPrsInRoom", "tellPrsInfoAtLocToPrsAtLoc", "followPrsAtLoc"
    ]
    person_cmd_list = ["goToLoc", "findPrsInRoom", "meetPrsAtBeac", "countPrsInRoom", "tellPrsInfoInLoc",
                           "talkInfoToGestPrsInRoom", "answerToGestPrsInRoom", "followNameFromBeacToRoom",
                           "guideNameFromBeacToBeac", "guidePrsFromBeacToBeac", "guideClothPrsFromBeacToBeac",
                           "greetClothDscInRm", "greetNameInRm", "meetNameAtLocThenFindInRm", "countClothPrsInRoom",
                           "countClothPrsInRoom", "tellPrsInfoAtLocToPrsAtLoc", "followPrsAtLoc"]
    
    # Object manipulation and perception commands
    object_cmd_list = ["goToLoc", "takeObjFromPlcmt", "findObjInRoom", "countObjOnPlcmt", "tellObjPropOnPlcmt",
                           "bringMeObjFromPlcmt", "tellCatPropOnPlcmt"]

    prep_dict = {
        "deliverPrep": ["to"],
        "placePrep": ["on"],
        "inLocPrep": ["in"],
        "fromLocPrep": ["from"],
        "toLocPrep": ["to"],
        "atLocPrep": ["at"],
        "talkPrep": ["to"],
        "locPrep": ["in", "at"],
        "onLocPrep": ["on"],
        "arePrep": ["are"],
        "ofPrsPrep": ["of"]
    }

    connector_list = ["and"]
    gesture_person_list = ["waving person", "person raising their left arm", "person raising their right arm",
                           "person pointing to the left", "person pointing to the right"]
    pose_person_list = ["sitting person", "standing person", "lying person"]
    gesture_person_plural_list = ["waving persons", "persons raising their left arm", "persons raising their right arm",
                                  "persons pointing to the left", "persons pointing to the right"]
    pose_person_plural_list = ["sitting persons", "standing persons", "lying persons"]

    person_info_list = ["name", "pose", "gesture"]
    object_comp_list = ["biggest", "largest", "smallest", "heaviest", "lightest", "thinnest"]

    talk_list = ["something about yourself", "the time", "what day is today", "what day is tomorrow", "your teams name",
                 "your teams country", "your teams affiliation", "the day of the week", "the day of the month"]
    question_list = ["question", "quiz"]

    color_list = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
    clothe_list = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]
    clothes_list = ["t shirts", "shirts", "blouses", "sweaters", "coats", "jackets"]
    color_clothe_list = [f"{a} {b}" for a, b in itertools.product(color_list, clothe_list)]
    color_clothes_list = [f"{a} {b}" for a, b in itertools.product(color_list, clothes_list)]
    
    def pick_unique(self, name_list, used_str):
        return random.choice([x for x in name_list if x not in used_str]) if name_list else "UNKNOWN"

    # ------------------------------
    # 1. Generate a common command context.
    # ------------------------------
    def generate_command_context(self, cmd_type=None):
        """
        Selects a command type and creates a dictionary with randomly chosen elements.
        This context is then used by both the string and structured generators.
        """
        
        if not cmd_type:
            cmd_type = random.choice(self.all_cmd_types)

        context = {"cmd_type": cmd_type}

        if cmd_type == "goToLoc":
            context["goVerb"] = self.insert_placeholders("goVerb")
            context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            context["loc_room"] = self.insert_placeholders("loc_room")
            return self.generate_command_follow_up_context('atLoc', [context])
    
        elif cmd_type == "takeObjFromPlcmt":
            context["takeVerb"] = self.insert_placeholders("takeVerb")
            context["art"] = self.insert_placeholders("art")
            context["obj_singCat"] = self.insert_placeholders("obj_singCat")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["plcmtLoc"] = self.insert_placeholders("plcmtLoc")
            return self.generate_command_follow_up_context('hasObj', [context])
    
        elif cmd_type == "findPrsInRoom":
            context["findVerb"] = self.insert_placeholders("findVerb")
            context["gestPers_posePers"] = self.insert_placeholders("gestPers_posePers")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return self.generate_command_follow_up_context('foundPers', [context])
    
        elif cmd_type == "findObjInRoom":
            context["findVerb"] = self.insert_placeholders("findVerb")
            context["art"] = self.insert_placeholders("art")
            context["obj_singCat"] = self.insert_placeholders("obj_singCat")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return self.generate_command_follow_up_context("foundObj", [context])
    
        elif cmd_type == "meetPrsAtBeac":
            context["meetVerb"] = self.insert_placeholders("meetVerb")
            context["name"] = self.insert_placeholders("name")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return self.generate_command_follow_up_context('foundPers', [context])
    
        elif cmd_type == "countObjOnPlcmt":
            context["countVerb"] = self.insert_placeholders("countVerb")
            context["plurCat"] = self.insert_placeholders("plurCat")
            context["onLocPrep"] = self.insert_placeholders("onLocPrep")
            context["plcmtLoc"] = self.insert_placeholders("plcmtLoc")
            return [context]
    
        elif cmd_type == "countPrsInRoom":
            context["countVerb"] = self.insert_placeholders("countVerb")
            context["gestPersPlur_posePersPlur"] = self.insert_placeholders("gestPersPlur_posePersPlur")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return [context]
    
        elif cmd_type == "tellPrsInfoInLoc":
            context["tellVerb"] = self.insert_placeholders("tellVerb")
            context["persInfo"] = self.insert_placeholders("persInfo")
            context["inRoom_atLoc"] = self.insert_placeholders("inRoom_atLoc")
            return [context]
    
        elif cmd_type == "tellObjPropOnPlcmt":
            context["tellVerb"] = self.insert_placeholders("tellVerb")
            context["objComp"] = self.insert_placeholders("objComp")
            context["onLocPrep"] = self.insert_placeholders("onLocPrep")
            context["plcmtLoc"] = self.insert_placeholders("plcmtLoc")
            return [context]
    
        elif cmd_type == "talkInfoToGestPrsInRoom":
            context["talkVerb"] = self.insert_placeholders("talkVerb")
            context["talk"] = self.insert_placeholders("talk")
            context["talkPrep"] = self.insert_placeholders("talkPrep")
            context["gestPers"] = self.insert_placeholders("gestPers")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return [context]
    
        elif cmd_type == "answerToGestPrsInRoom":
            context["answerVerb"] = self.insert_placeholders("answerVerb")
            context["question"] = self.insert_placeholders("question")
            context["ofPrsPrep"] = self.insert_placeholders("ofPrsPrep")
            context["gestPers"] = self.insert_placeholders("gestPers")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return [context]
    
        elif cmd_type == "followNameFromBeacToRoom":
            context["followVerb"] = self.insert_placeholders("followVerb")
            context["name"] = self.insert_placeholders("name")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["loc"] = self.insert_placeholders("loc")
            context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            context["room"] = self.insert_placeholders("room")
            return [context]
    
        elif cmd_type == "guideNameFromBeacToBeac":
            context["guideVerb"] = self.insert_placeholders("guideVerb")
            context["name"] = self.insert_placeholders("name")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["loc"] = self.insert_placeholders("loc")
            context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            context["loc_room"] = self.insert_placeholders("loc_room")
            return [context]
    
        elif cmd_type == "guidePrsFromBeacToBeac":
            context["guideVerb"] = self.insert_placeholders("guideVerb")
            context["gestPers_posePers"] = self.insert_placeholders("gestPers_posePers")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["loc"] = self.insert_placeholders("loc")
            context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            context["loc_room"] = self.insert_placeholders("loc_room")
            return [context]
    
        elif cmd_type == "guideClothPrsFromBeacToBeac":
            context["guideVerb"] = self.insert_placeholders("guideVerb")
            context["colorClothe"] = self.insert_placeholders("colorClothe")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["loc"] = self.insert_placeholders("loc")
            context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            context["loc_room"] = self.insert_placeholders("loc_room")
            return [context]
    
        elif cmd_type == "bringMeObjFromPlcmt":
            context["bringVerb"] = self.insert_placeholders("bringVerb")
            context["art"] = self.insert_placeholders("art")
            context["obj"] = self.insert_placeholders("obj")
            context["fromLocPrep"] = self.insert_placeholders("fromLocPrep")
            context["plcmtLoc"] = self.insert_placeholders("plcmtLoc")
            return [context]
    
        elif cmd_type == "tellCatPropOnPlcmt":
            context["tellVerb"] = self.insert_placeholders("tellVerb")
            context["objComp"] = self.insert_placeholders("objComp")
            context["singCat"] = self.insert_placeholders("singCat")
            context["onLocPrep"] = self.insert_placeholders("onLocPrep")
            context["plcmtLoc"] = self.insert_placeholders("plcmtLoc")
            return [context]
    
        elif cmd_type == "greetClothDscInRm":
            context["greetVerb"] = self.insert_placeholders("greetVerb")
            context["art"] = self.insert_placeholders("art")
            context["colorClothe"] = self.insert_placeholders("colorClothe")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return self.generate_command_follow_up_context("foundPers", [context])
    
        elif cmd_type == "greetNameInRm":
            context["greetVerb"] = self.insert_placeholders("greetVerb")
            context["name"] = self.insert_placeholders("name")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return self.generate_command_follow_up_context("foundPers", [context])
    
        elif cmd_type == "meetNameAtLocThenFindInRm":
            context["meetVerb"] = self.insert_placeholders("meetVerb")
            context["name"] = self.insert_placeholders("name")
            context["atLocPrep"] = self.insert_placeholders("atLocPrep")
            context["loc"] = self.insert_placeholders("loc")
            context["findVerb"] = self.insert_placeholders("findVerb")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            return [context]
    
        elif cmd_type == "countClothPrsInRoom":
            context["countVerb"] = self.insert_placeholders("countVerb")
            context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            context["room"] = self.insert_placeholders("room")
            context["colorClothes"] = self.insert_placeholders("colorClothes")
            return [context]
    
        elif cmd_type == "tellPrsInfoAtLocToPrsAtLoc":
            context["tellVerb"] = self.insert_placeholders("tellVerb")
            context["persInfo"] = self.insert_placeholders("persInfo")
            context["atLocPrep"] = self.insert_placeholders("atLocPrep")
            context["loc"] = self.insert_placeholders("loc")

            context["loc2"] = self.pick_unique(self.location_names, str(context))
            return [context]
    
        elif cmd_type == "followPrsAtLoc":
            context["followVerb"] = self.insert_placeholders("followVerb")
            context["gestPers_posePers"] = self.insert_placeholders("gestPers_posePers")
            context["inRoom_atLoc"] = self.insert_placeholders("inRoom_atLoc")
            return [context]
    
    def generate_command_follow_up_context(self, cmd_type, context: list[dict]):
        cmd_category = ""
        if cmd_type in self.person_cmd_list and cmd_type in self.object_cmd_list:
            cmd_category = "both"
        elif cmd_type in self.person_cmd_list:
            cmd_category = "people"
        else:
            cmd_category = "objects"
    
        follow_up_context = {}
    
        # Determine follow-up command
        if cmd_type == "atLoc":
            if cmd_category == "people":
                follow_up_cmd = random.choice(["findPrs", "meetName"])
            elif cmd_category == "objects":
                follow_up_cmd = random.choice(["findObj"])
            else:
                follow_up_cmd = random.choice(["findPrs", "meetName", "findObj"])
        elif cmd_type == "hasObj":
            follow_up_cmd = random.choice([
                "placeObjOnPlcmt", "deliverObjToMe",
                "deliverObjToPrsInRoom", "deliverObjToNameAtBeac"
            ])
        elif cmd_type == "foundPers":
            follow_up_cmd = random.choice([
                "talkInfo", "answerQuestion",
                "followPrs", "followPrsToRoom", "guidePrsToBeacon"
            ])
        elif cmd_type == "foundObj":
            follow_up_cmd = random.choice(["takeObj"])
        else:
            raise ValueError("Invalid command type for follow-up context generation: " + cmd_type)
    
        follow_up_context["cmd_type"] = follow_up_cmd
    
        # Avoid repeated locations
        context_str = str(context)
    
    
        # Fill placeholders
        if follow_up_cmd == "findObj":
            follow_up_context["findVerb"] = self.insert_placeholders("findVerb")
            follow_up_context["art"] = self.insert_placeholders("art")
            follow_up_context["obj_singCat"] = random.choice(self.object_names + self.object_categories_singular)
            return self.generate_command_follow_up_context("foundObj", context + [follow_up_context])
    
        elif follow_up_cmd == "findPrs":
            follow_up_context["findVerb"] = self.insert_placeholders("findVerb")
            follow_up_context["gestPers_posePers"] = self.insert_placeholders("gestPers_posePers")
            return self.generate_command_follow_up_context("foundPers", context + [follow_up_context])
    
        elif follow_up_cmd == "meetName":
            follow_up_context["meetVerb"] = self.insert_placeholders("meetVerb")
            follow_up_context["name"] = self.insert_placeholders("name")
            return self.generate_command_follow_up_context("foundPers", context + [follow_up_context])
    
        elif follow_up_cmd == "placeObjOnPlcmt":
            follow_up_context["placeVerb"] = self.insert_placeholders("placeVerb")
            follow_up_context["onLocPrep"] = self.insert_placeholders("onLocPrep")
            follow_up_context["plcmtLoc2"] = self.pick_unique(self.placement_location_names, context_str)
    
        elif follow_up_cmd == "deliverObjToMe":
            follow_up_context["deliverVerb"] = self.insert_placeholders("deliverVerb")
    
        elif follow_up_cmd == "deliverObjToPrsInRoom":
            follow_up_context["deliverVerb"] = self.insert_placeholders("deliverVerb")
            follow_up_context["deliverPrep"] = self.insert_placeholders("deliverPrep")
            follow_up_context["gestPers_posePers"] = self.insert_placeholders("gestPers_posePers")
            follow_up_context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            follow_up_context["room"] = self.pick_unique(self.room_names, context_str)
    
        elif follow_up_cmd == "deliverObjToNameAtBeac":
            follow_up_context["deliverVerb"] = self.insert_placeholders("deliverVerb")
            follow_up_context["deliverPrep"] = self.insert_placeholders("deliverPrep")
            follow_up_context["name"] = self.insert_placeholders("name")
            follow_up_context["inLocPrep"] = self.insert_placeholders("inLocPrep")
            follow_up_context["room"] = self.pick_unique(self.room_names, context_str)
    
        elif follow_up_cmd == "talkInfo":
            follow_up_context["talkVerb"] = self.insert_placeholders("talkVerb")
            follow_up_context["talk"] = self.insert_placeholders("talk")
    
        elif follow_up_cmd == "answerQuestion":
            follow_up_context["answerVerb"] = self.insert_placeholders("answerVerb")
            follow_up_context["question"] = self.insert_placeholders("question")
    
        elif follow_up_cmd == "followPrs":
            follow_up_context["followVerb"] = self.insert_placeholders("followVerb")
    
        elif follow_up_cmd == "followPrsToRoom":
            follow_up_context["followVerb"] = self.insert_placeholders("followVerb")
            follow_up_context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            follow_up_context["loc2_room2"] = self.pick_unique(self.room_names + self.location_names, context_str)
    
        elif follow_up_cmd == "guidePrsToBeacon":
            follow_up_context["guideVerb"] = self.insert_placeholders("guideVerb")
            follow_up_context["toLocPrep"] = self.insert_placeholders("toLocPrep")
            follow_up_context["loc2_room2"] = self.pick_unique(self.room_names + self.location_names, context_str)
    
        elif follow_up_cmd == "takeObj":
            follow_up_context["takeVerb"] = self.insert_placeholders("takeVerb")
            return self.generate_command_follow_up_context("hasObj", context + [follow_up_context])
    
        return context + [follow_up_context]
    
    
    # ------------------------------
    # 2. Generate a one-line string command from the context.
    # ------------------------------
    def generate_command_string(self, context):
        
        cur_context = context[0]
        
        command = cur_context["cmd_type"]
        if command == "goToLoc":
            return f"{cur_context['goVerb']} {cur_context['toLocPrep']} the {cur_context['loc_room']} then " + self.generate_command_followup(context[1:])
        elif command == "takeObjFromPlcmt":
            return f"{cur_context['takeVerb']} {cur_context['art']} {cur_context['obj_singCat']} {cur_context['fromLocPrep']} the {cur_context['plcmtLoc']} and " + \
                                self.generate_command_followup(context[1:])
        elif command == "findPrsInRoom":
            return f"{cur_context['findVerb']} a {cur_context['gestPers_posePers']} {cur_context['inLocPrep']} the {cur_context['room']} and " + \
                                self.generate_command_followup(context[1:])
        elif command == "findObjInRoom":
            return f"{cur_context['findVerb']} {cur_context['art']} {cur_context['obj_singCat']} {cur_context['inLocPrep']} the {cur_context['room']} then " + \
                                self.generate_command_followup(context[1:])
        elif command == "meetPrsAtBeac":
            return f"{cur_context['meetVerb']} {cur_context['name']} {cur_context['inLocPrep']} the {cur_context['room']} and " + \
                                self.generate_command_followup(context[1:])
        elif command == "countObjOnPlcmt":
            return f"{cur_context['countVerb']} {cur_context['plurCat']} there are {cur_context['onLocPrep']} the {cur_context['plcmtLoc']}"
        elif command == "countPrsInRoom":
            return f"{cur_context['countVerb']} {cur_context['gestPersPlur_posePersPlur']} are {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == "tellPrsInfoInLoc":
            return f"{cur_context['tellVerb']} me the {cur_context['persInfo']} of the person {cur_context['inRoom_atLoc']}"
        elif command == "tellObjPropOnPlcmt":
            return f"{cur_context['tellVerb']} me what is the {cur_context['objComp']} object {cur_context['onLocPrep']} the {cur_context['plcmtLoc']}"
        elif command == "talkInfoToGestPrsInRoom":
            return f"{cur_context['talkVerb']} {cur_context['talk']} {cur_context['talkPrep']} the {cur_context['gestPers']} {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == "answerToGestPrsInRoom":
            return f"{cur_context['answerVerb']} the {cur_context['question']} {cur_context['ofPrsPrep']} the {cur_context['gestPers']} {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == "followNameFromBeacToRoom":
            return f"{cur_context['followVerb']} {cur_context['name']} {cur_context['fromLocPrep']} the {cur_context['loc']} {cur_context['toLocPrep']} the {cur_context['room']}"
        elif command == "guideNameFromBeacToBeac":
            return f"{cur_context['guideVerb']} {cur_context['name']} {cur_context['fromLocPrep']} the {cur_context['loc']} {cur_context['toLocPrep']} the {cur_context['loc_room']}"
        elif command == "guidePrsFromBeacToBeac":
            return f"{cur_context['guideVerb']} the {cur_context['gestPers_posePers']} {cur_context['fromLocPrep']} the {cur_context['loc']} {cur_context['toLocPrep']} the {cur_context['loc_room']}"
        elif command == "guideClothPrsFromBeacToBeac":
            return f"{cur_context['guideVerb']} the person wearing a {cur_context['colorClothe']} {cur_context['fromLocPrep']} the {cur_context['loc']} {cur_context['toLocPrep']} the {cur_context['loc_room']}"
        elif command == "bringMeObjFromPlcmt":
            return f"{cur_context['bringVerb']} me {cur_context['art']} {cur_context['obj']} {cur_context['fromLocPrep']} the {cur_context['plcmtLoc']}"
        elif command == "tellCatPropOnPlcmt":
            return f"{cur_context['tellVerb']} me what is the {cur_context['objComp']} {cur_context['singCat']} {cur_context['onLocPrep']} the {cur_context['plcmtLoc']}"
        elif command == "greetClothDscInRm":
            return f"{cur_context['greetVerb']} the person wearing {cur_context['art']} {cur_context['colorClothe']} {cur_context['inLocPrep']} the {cur_context['room']} and " + \
                   self.generate_command_followup(context[1:])
        elif command == "greetNameInRm":
            return f"{cur_context['greetVerb']} {cur_context['name']} {cur_context['inLocPrep']} the {cur_context['room']} and " + \
                   self.generate_command_followup(context[1:])
        elif command == "meetNameAtLocThenFindInRm":
            return f"{cur_context['meetVerb']} {cur_context['name']} {cur_context['atLocPrep']} the {cur_context['loc']} then {cur_context['findVerb']} them {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == "countClothPrsInRoom":
            return f"{cur_context['countVerb']} people {cur_context['inLocPrep']} the {cur_context['room']} are wearing {cur_context['colorClothes']}"
        elif command == "tellPrsInfoAtLocToPrsAtLoc":
            return f"{cur_context['tellVerb']} the {cur_context['persInfo']} of the person {cur_context['atLocPrep']} the {cur_context['loc']} to the person {cur_context['atLocPrep']} the {cur_context['loc2']}"
        elif command == "followPrsAtLoc":
            return f"{cur_context['followVerb']} the {cur_context['gestPers_posePers']} {cur_context['inRoom_atLoc']}"
        else:
            warnings.warn("Command type not covered: " + command)
            return "WARNING"


    def generate_command_followup(self, context):
        cur_context = context[0]
        command = cur_context["cmd_type"]
        if command == "findObj":
            return f"{cur_context['findVerb']} {cur_context['art']} {cur_context['obj_singCat']} and " + \
                             self.generate_command_followup(context[1:])
        elif command == 'findPrs':
            return f"{cur_context['findVerb']} the {cur_context['gestPers_posePers']} and " + \
                             self.generate_command_followup( context[1:])
        elif command == 'meetName':
            return f"{cur_context['meetVerb']} {cur_context['name']} and " + \
                             self.generate_command_followup(context[1:])
        elif command == 'placeObjOnPlcmt':
            return f"{cur_context['placeVerb']} it {cur_context['onLocPrep']} the {cur_context['plcmtLoc2']}"
        elif command == 'deliverObjToMe':
            return f"{cur_context['deliverVerb']} it to me"
        elif command == 'deliverObjToPrsInRoom':
            return f"{cur_context['deliverVerb']} it {cur_context['deliverPrep']} the {cur_context['gestPers_posePers']} {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == 'deliverObjToNameAtBeac':
            return f"{cur_context['deliverVerb']} it {cur_context['deliverPrep']} {cur_context['name']} {cur_context['inLocPrep']} the {cur_context['room']}"
        elif command == 'talkInfo':
            return f"{cur_context['talkVerb']} {cur_context['talk']}"
        elif command == 'answerQuestion':
            return f"{cur_context['answerVerb']} a {cur_context['question']}"
        elif command == 'followPrs':
            return f"{cur_context['followVerb']} them"
        elif command == 'followPrsToRoom':
            return f"{cur_context['followVerb']} them {cur_context['toLocPrep']} the {cur_context['loc2_room2']}"
        elif command == 'guidePrsToBeacon':
            return f"{cur_context['guideVerb']} them {cur_context['toLocPrep']} the {cur_context['loc2_room2']}"
        elif command == 'takeObj':
            return f"{cur_context['takeVerb']} it and " + \
                             self.generate_command_followup(context[1:])
        else:
            warnings.warn("Command type not covered: " + command)
            return "WARNING"

    # ------------------------------
    # 3. Generate structured command steps from the same context.
    # ------------------------------
    def generate_structured_command(self, context, context_index=0):
        cur_context = context[context_index]
        context_index += 1
        ct = cur_context["cmd_type"]
        steps = []
        
        if ct == "goToLoc":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc_room']})
        elif ct == "takeObjFromPlcmt":
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc']})
            # steps.append({"action": "find_object", "area_to_search": cur_context['plcmtLoc'], "object_to_find": cur_context['obj_singCat']})
            steps.append({"action": "pick_object", "object_to_pick": cur_context['obj_singCat']})
        elif ct == "findPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers_posePers']})
        elif ct == "findObjInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            # steps.append({"action": "find_object", "area_to_search": cur_context['room'], "object_to_find": cur_context['obj_singCat']})
        elif ct == "meetPrsAtBeac":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
        elif ct == "countObjOnPlcmt":
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc']})
            steps.append({"action": "count", "target_to_count": cur_context['plurCat']})
            steps.append({"action": "go_to", "location_to_go": 'start_location'})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "count"})
        elif ct == "countPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "count", "target_to_count": cur_context['gestPersPlur_posePersPlur']})
            steps.append({"action": "go_to", "location_to_go": 'start_location'})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "count"})
        elif ct == "tellPrsInfoInLoc":
            steps.append({"action": "go_to", "location_to_go": cur_context['inRoom_atLoc'].split("the ", 1)[-1]})
            steps.append({"action": "find_person", "attribute_value": ''})
            steps.append({"action": "get_person_info", "info_type": cur_context['persInfo']})
            steps.append({"action": "go_to", "location_to_go": 'start_location'})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "get_person_info"})
        elif ct == "tellObjPropOnPlcmt": 
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc']})
            steps.append({"action": "get_visual_info", "measure": cur_context['objComp'], "object_category": "object"})
            steps.append({"action": "go_to", "location_to_go": 'start_location'})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "get_visual_info"})
        elif ct == "talkInfoToGestPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers']})
            steps.append({"action": "say_with_context", "user_instruction":  self.sub_art(self.generate_command_string(context)), "previous_command_info": cur_context['talk']})
        elif ct == "answerToGestPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers']})
            steps.append({"action": "answer_question"})
        elif ct == "followNameFromBeacToRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
            steps.append({"action": "follow_person_until", "destination": cur_context['room']})            
        elif ct == "guideNameFromBeacToBeac":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
            steps.append({"action": "guide_person_to", "destination_room": cur_context['loc_room']})
            
        elif ct == "guidePrsFromBeacToBeac":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers_posePers']})
            steps.append({"action": "guide_person_to", "destination_room": cur_context['loc_room']})
        elif ct == "guideClothPrsFromBeacToBeac":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person", "attribute_value": cur_context['colorClothe']})
            steps.append({"action": "guide_person_to", "destination_room": cur_context['loc_room']})
        elif ct == "bringMeObjFromPlcmt":
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc']})
            # steps.append({"action": "find_object", "area_to_search": cur_context['plcmtLoc'], "object_to_find": cur_context['obj']})
            steps.append({"action": "pick_object", "object_to_pick": cur_context['obj']})
            steps.append({"action": "go_to", "location_to_go": "start_location"})
            steps.append({"action": "give_object"})
        elif ct == "tellCatPropOnPlcmt":
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc']})
            steps.append({"action": "get_visual_info", "measure": cur_context['objComp'], "object_category": cur_context['singCat']})
            steps.append({"action": "go_to", "location_to_go": "start_location"})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "get_visual_info"})
        elif ct == "greetClothDscInRm":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person", "attribute_value": cur_context['colorClothe']})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "introduction"})
        elif ct == "greetNameInRm":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "introduction"})
        elif ct == "meetNameAtLocThenFindInRm":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
        elif ct == "countClothPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "count", "target_to_count": f"people wearing {cur_context['colorClothes']}"})
            steps.append({"action": "go_to", "location_to_go": "start_location"})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "count"})
        elif ct == "tellPrsInfoAtLocToPrsAtLoc":
            steps.append({"action": "go_to", "location_to_go": cur_context['loc']})
            steps.append({"action": "find_person", "attribute_value": ""})
            steps.append({"action": "get_person_info", "info_type": cur_context['persInfo']})
            steps.append({"action": "go_to", "location_to_go": cur_context['loc2']})
            steps.append({"action": "find_person", "attribute_value": ""})
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": "get_person_info"})
        elif ct == "followPrsAtLoc":
            steps.append({"action": "go_to", "location_to_go": cur_context['inRoom_atLoc'].split("the ", 1)[-1]})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers_posePers']})
            steps.append({"action": "follow_person_until", "destination": "canceled"})
        elif ct == "findObj":
            pass
            # steps.append({"action": "find_object", "area_to_search": self.find_in_context('loc_room', context, context_index-1), "object_to_find":  cur_context['obj_singCat']})
        elif ct == "findPrs":
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers_posePers']})
        elif ct == "meetName":
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
        elif ct == "placeObjOnPlcmt":
            steps.append({"action": "go_to", "location_to_go": cur_context['plcmtLoc2']})
            steps.append({"action": "place_object"}) 
        elif ct == "deliverObjToMe":
            steps.append({"action": "go_to", "location_to_go": "start_location"})
            steps.append({"action": "give_object"})
        elif ct == "deliverObjToPrsInRoom":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person", "attribute_value": cur_context['gestPers_posePers']})
            steps.append({"action": "give_object"})
        elif ct == "deliverObjToNameAtBeac":
            steps.append({"action": "go_to", "location_to_go": cur_context['room']})
            steps.append({"action": "find_person_by_name", "name": cur_context['name']})
            steps.append({"action": "give_object"})
        elif ct == "talkInfo":
            steps.append({"action": "say_with_context", "user_instruction": self.sub_art(self.generate_command_string(context)), "previous_command_info": cur_context['talk']})
        elif ct == "answerQuestion":
            steps.append({"action": "answer_question"})
        elif ct == "followPrs":
            steps.append({"action": "follow_person_until", "destination": "canceled"})
        elif ct == "followPrsToRoom":
            steps.append({"action": "follow_person_until", "destination": cur_context['loc2_room2']})
        elif ct == "guidePrsToBeacon":
            steps.append({"action": "guide_person_to", "destination_room": cur_context['loc2_room2']})
        elif ct == "takeObj":
            steps.append({"action": "pick_object", "object_to_pick": self.find_in_context('obj_singCat', context, context_index-1)})
        else:
            raise Exception(f"Command type not implemented: {ct}")
        
        if len(context) == context_index:
            return steps
        
        return steps + self.generate_structured_command(context, context_index)

    def find_in_context(self, target: str, context: list[dict], context_index: int):
        for i in range(context_index, -1, -1):
            if target in context[i]:
                return context[i][target]
        
        raise ValueError(f"ERROR, {target} not found in context.")
    
    def sub_art(self, command_string):
        art_ph = re.findall(r'\{(art)\}\s*([A-Za-z])', command_string, re.DOTALL)
        if art_ph:
            command_string = command_string.replace("art",
                                                    "an" if art_ph[0][1].lower() in ["a", "e", "i", "o", "u"] else "a")
        return command_string.replace('{', '').replace('}', '')
    # ------------------------------
    # 4. Generate both commands using the common context.
    # ------------------------------
    def generate_full_command(self, cmd_type="", difficulty=0):
        context = self.generate_command_context(cmd_type)
        string_command = self.sub_art(self.generate_command_string(context))
        structured_command = self.generate_structured_command(context)
        return string_command, structured_command

    def insert_placeholders(self, ph):
        ph = ph.replace('{', '').replace('}', '')
        if len(ph.split('_')) > 1:
            ph = random.choice(ph.split('_'))
        if ph == "goVerb":
            return random.choice(self.verb_dict["go"])
        elif ph == "takeVerb":
            return random.choice(self.verb_dict["take"])
        elif ph == "findVerb":
            return random.choice(self.verb_dict["find"])
        elif ph == "meetVerb":
            return random.choice(self.verb_dict["meet"])
        elif ph == "countVerb":
            return random.choice(self.verb_dict["count"])
        elif ph == "tellVerb":
            return random.choice(self.verb_dict["tell"])
        elif ph == "deliverVerb":
            return random.choice(self.verb_dict["deliver"])
        elif ph == "talkVerb":
            return random.choice(self.verb_dict["talk"])
        elif ph == "answerVerb":
            return random.choice(self.verb_dict["answer"])
        elif ph == "followVerb":
            return random.choice(self.verb_dict["follow"])
        elif ph == "placeVerb":
            return random.choice(self.verb_dict["place"])
        elif ph == "guideVerb":
            return random.choice(self.verb_dict["guide"])
        elif ph == "greetVerb":
            return random.choice(self.verb_dict["greet"])
        elif ph == "bringVerb":
            return random.choice(self.verb_dict["bring"])

        elif ph == "toLocPrep":
            return random.choice(self.prep_dict["toLocPrep"])
        elif ph == "fromLocPrep":
            return random.choice(self.prep_dict["fromLocPrep"])
        elif ph == "inLocPrep":
            return random.choice(self.prep_dict["inLocPrep"])
        elif ph == "onLocPrep":
            return random.choice(self.prep_dict["onLocPrep"])
        elif ph == "atLocPrep":
            return random.choice(self.prep_dict["atLocPrep"])
        elif ph == "deliverPrep":
            return random.choice(self.prep_dict["deliverPrep"])
        elif ph == "talkPrep":
            return random.choice(self.prep_dict["talkPrep"])
        elif ph == "ofPrsPrep":
            return random.choice(self.prep_dict["ofPrsPrep"])

        elif ph == "connector":
            return random.choice(self.connector_list)

        elif ph == "plcmtLoc2":
            return "plcmtLoc2"
        elif ph == "plcmtLoc":
            return random.choice(self.placement_location_names)
        elif ph == "room2":
            return "room2"
        elif ph == "room":
            return random.choice(self.room_names)
        elif ph == "loc2":
            return "loc2"
        elif ph == "loc":
            return random.choice(self.location_names)
        elif ph == "inRoom":
            return random.choice(self.prep_dict["inLocPrep"]) + " the " + random.choice(self.room_names)
        elif ph == "atLoc":
            return random.choice(self.prep_dict["atLocPrep"]) + " the " + random.choice(self.location_names)

        elif ph == "gestPers":
            return random.choice(self.gesture_person_list)
        elif ph == "posePers":
            return random.choice(self.pose_person_list)
        elif ph == "name":
            return random.choice(self.person_names)
        elif ph == "gestPersPlur":
            return random.choice(self.gesture_person_plural_list)
        elif ph == "posePersPlur":
            return random.choice(self.pose_person_plural_list)
        elif ph == "persInfo":
            return random.choice(self.person_info_list)

        elif ph == "obj":
            return random.choice(self.object_names)
        elif ph == "singCat":
            return random.choice(self.object_categories_singular)
        elif ph == "plurCat":
            return random.choice(self.object_categories_plural)
        elif ph == "objComp":
            return random.choice(self.object_comp_list)

        elif ph == "talk":
            return random.choice(self.talk_list)
        elif ph == "question":
            return random.choice(self.question_list)

        elif ph == "colorClothe":
            return random.choice(self.color_clothe_list)
        elif ph == "colorClothes":
            return random.choice(self.color_clothes_list)

        # replace article later
        elif ph == "art":
            return "{art}"
        else:
            warnings.warn("Placeholder not covered: " + ph)
            return "WARNING"
# ------------------------------
# EXAMPLE USAGE:
# ------------------------------
if __name__ == "__main__":
    # Provide dummy data.
    person_names = ["Axel", "Jane", "Morgan", "Adel"]
    location_names = ["kitchen", "office", "pantry", "bedroom", "living_room", "bathroom"]
    placement_location_names = ["dishwasher", "desk", "trashbin", "side_tables", "bookshelf", "kitchen_table"]
    room_names = ["kitchen", "bedroom", "living_room", "bathroom", "entrance"]
    object_names = ["apple", "snack", "soccer_ball", "cleaning_supply", "toy"]
    object_categories_plural = ["fruits", "toys"]
    object_categories_singular = ["fruit", "toy"]

    cg = CommandGenerator(person_names, location_names, placement_location_names, room_names, object_names,
                          object_categories_plural, object_categories_singular)

    # Generate both the string command and the structured command.
    string_cmd, structured_cmd = cg.generate_full_command(cmd_category="")
    print("String Command:")
    print(string_cmd)
    print("\nStructured Command Steps:")
    for step in structured_cmd:
        print(step)
