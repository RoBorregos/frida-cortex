from baml_client.types import (
    AnswerQuestion,
    GetVisualInfo,
    GoTo,
    PickObject,
    PlaceObject,
    SayWithContext,
    GiveObject,
    FollowPersonUntil,
    GuidePersonTo,
    Count,
    GetPersonInfo,
    FindPerson,
    FindPersonByName
)
from baml_client.sync_client import b

from command_interpreter.status import Status
from command_interpreter.embeddings.categorization import Embeddings

class Tasks:
    def __init__(self):
        self.tasks = {
            "object_detection": "Detect objects in an image.",
            "image_classification": "Classify the content of an image.",
            "face_recognition": "Recognize faces in an image.",
            "image_segmentation": "Segment different parts of an image.",
            "optical_character_recognition": "Extract text from images."
        }
        self.embeddings = Embeddings()

    
    def go_to(self, command: GoTo, grounding: bool = True) -> tuple[Status, str]:
        if grounding:
            query_result = self.embeddings.query_location(command.location_to_go)
            area = self.embeddings.get_area(query_result)
            subarea = self.embeddings.get_subarea(query_result)
        else:
            area = command.location_to_go
            subarea = None
        return (Status.EXECUTION_SUCCESS, "arrived to: " +
                (area + (" -> " + subarea if subarea else "")))
    
    def pick_object(self, command: PickObject, grounding: bool = True) -> tuple[Status, str]:
        if grounding:
            query_result = self.embeddings.query_item(command.object_to_pick)
            name = self.embeddings.get_name(query_result)
        else:
            name = command.object_to_pick
        return (Status.EXECUTION_SUCCESS, "picked up: " + name)
    
    def place_object(self, command: PlaceObject, grounding: bool = True):
        return Status.EXECUTION_SUCCESS, "placed object"
    
    def say_with_context(self, command: SayWithContext, grounding: bool = True):
        if grounding:
            query_command_history = self.embeddings.query_command_history(command.previous_command_info + " " + command.user_instruction)
            query_tec_knowledge = self.embeddings.query_tec_knowledge(command.previous_command_info + " " + command.user_instruction)
            query_frida_knowledge = self.embeddings.query_frida_knowledge(command.previous_command_info + " " + command.user_instruction)
            query_roborregos_knowledge = self.embeddings.query_roborregos_knowledge(command.previous_command_info + " " + command.user_instruction)
            query_result = ("command history: " + str(query_command_history) + "\n"
                            + "tec knowledge: " + str(query_tec_knowledge) + "\n"
                            + "frida knowledge: " + str(query_frida_knowledge) + "\n"
                            + "roborregos knowledge: " + str(query_roborregos_knowledge))
        else:
            query_result = command.previous_command_info
        response = b.AugmentedResponse(query_result, command.user_instruction)
        return Status.EXECUTION_SUCCESS, response
    
    def answer_question(self, command: AnswerQuestion, grounding: bool = True):
        # It is assumed it always answers the question
        return Status.EXECUTION_SUCCESS, "answered user's question"
    
    def get_visual_info(self, command: GetVisualInfo, grounding: bool = True):
        # It is assumed it always finds a box as the desired object
        return Status.EXECUTION_SUCCESS, "found: box as " + command.measure + " " + command.object_category
    
    def give_object(self, command: GiveObject, grounding: bool = True):
        return Status.EXECUTION_SUCCESS, "object given"
    
    def follow_person_until(self, command: FollowPersonUntil, grounding: bool = True):
        if command.destination == "canceled" or command.destination == "cancelled":
            return Status.EXECUTION_SUCCESS, "followed user until canceled"
        if grounding:
            query_result = self.embeddings.query_location(command.destination)
            area = self.embeddings.get_area(query_result)
            subarea = self.embeddings.get_subarea(query_result)
        else:
            area = command.destination
            subarea = None
        return Status.EXECUTION_SUCCESS, "arrived to: " + (area + (" -> " + subarea if subarea else ""))
    
    def guide_person_to(self, command: GuidePersonTo, grounding: bool = True):
        if grounding:
            query_result = self.embeddings.query_location(command.destination_room)
            area = self.embeddings.get_area(query_result)
            subarea = self.embeddings.get_subarea(query_result)
        else:
            area = command.destination_room
            subarea = None
        return Status.EXECUTION_SUCCESS, "arrived to: " + (area + (" -> " + subarea if subarea else ""))

    def get_person_info(self, command: GetPersonInfo, grounding: bool = True):
        if command.info_type == "gesture":
            return Status.EXECUTION_SUCCESS, "person gesture is pointing to the right"
        elif command.info_type == "pose":
            return Status.EXECUTION_SUCCESS, "person pose is standing"
        elif command.info_type == "name":
            return Status.EXECUTION_SUCCESS, "person name is John"
        
        return Status.EXECUTION_SUCCESS, "person " + command.info_type + " was found"
    
    def count(self, command: Count, grounding: bool = True):
        # Always returns 4
        return Status.EXECUTION_SUCCESS, "found: 4 " + command.target_to_count
    
    def find_person(self, command: FindPerson, grounding: bool = True):
        # Is assumed it always finds the person
        if command.attribute_value == "":
            return Status.EXECUTION_SUCCESS, "found person"
        else:
            return Status.EXECUTION_SUCCESS, "found person with attribute: " + command.attribute_value
    
    def find_person_by_name(self, command: FindPersonByName, grounding: bool = True):
        # Is assumed it always finds the person
        return Status.EXECUTION_SUCCESS, f"found {command.name}"
    
    def add_command_history(self, command, res, status):
        self.embeddings.add_command_history(
            command,
            res,
            status,
        )

    def clear_command_history(self):
        """Clears the command history before execution"""
        self.embeddings.delete_collection("command_history")
        self.embeddings.build_embeddings()