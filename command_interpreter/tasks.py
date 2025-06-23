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

    
    def go_to(self, command: GoTo) -> tuple[Status, str]:
        query_result = self.embeddings.query_location(command.location_to_go)
        area = self.embeddings.get_area(query_result)
        subarea = self.embeddings.get_subarea(query_result)
        return (Status.EXECUTION_SUCCESS, "arrived to: " +
                (area + (" -> " + subarea if subarea else "")))
    
    def pick_object(self, command: PickObject) -> tuple[Status, str]:
        query_result = self.embeddings.query_item(command.object_to_pick)
        name = self.embeddings.get_name(query_result)
        return (Status.EXECUTION_SUCCESS, "picked up: " + name)
    
    def place_object(self, command: PlaceObject):
        return Status.EXECUTION_SUCCESS, "placed object"
    
    def say_with_context(self, command: SayWithContext):
        query_result = self.embeddings.query_command_history(command.previous_command_info)
        response = b.AugmentedResponse(str(query_result), command.user_instruction)
        return Status.EXECUTION_SUCCESS, response
    
    def answer_question(self, command: AnswerQuestion):
        # It is assumed it always answers the question
        return Status.EXECUTION_SUCCESS, "answered user's question"
    
    def get_visual_info(self, command: GetVisualInfo):
        # It is assumed it always finds a box as the desired object
        return Status.EXECUTION_SUCCESS, "found: box as " + command.measure + " " + command.object_category
    
    def give_object(self, command: GiveObject):
        return Status.EXECUTION_SUCCESS, "object given"
    
    def follow_person_until(self, command: FollowPersonUntil):
        if command.destination == "canceled" or command.destination == "cancelled":
            return Status.EXECUTION_SUCCESS, "followed user until canceled"
        query_result = self.embeddings.query_location(command.destination)
        area = self.embeddings.get_area(query_result)
        subarea = self.embeddings.get_subarea(query_result)
        return Status.EXECUTION_SUCCESS, "arrived to: " + (area + (" -> " + subarea if subarea else ""))
    
    def guide_person_to(self, command: GuidePersonTo):
        query_result = self.embeddings.query_location(command.destination_room)
        area = self.embeddings.get_area(query_result)
        subarea = self.embeddings.get_subarea(query_result)
        return Status.EXECUTION_SUCCESS, "arrived to: " + (area + (" -> " + subarea if subarea else ""))

    def get_person_info(self, command: GetPersonInfo):
        if command.info_type == "gesture":
            return Status.EXECUTION_SUCCESS, "person gesture is pointing to the right"
        elif command.info_type == "pose":
            return Status.EXECUTION_SUCCESS, "person pose is standing"
        elif command.info_type == "name":
            return Status.EXECUTION_SUCCESS, "person name is John"
        
        return Status.EXECUTION_SUCCESS, "person " + command.info_type + " was found"
    
    def count(self, command: Count):
        # Always returns 4
        return Status.EXECUTION_SUCCESS, "found: 4 " + command.target_to_count
    
    def find_person(self, command: FindPerson):
        # Is assumed it always finds the person
        if command.attribute_value == "":
            return Status.EXECUTION_SUCCESS, "found person"
        else:
            return Status.EXECUTION_SUCCESS, "found person with attribute: " + command.attribute_value
    
    def find_person_by_name(self, command: FindPersonByName):
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