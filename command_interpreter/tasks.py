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
    FindPersonByName
)

from status import Status

class Tasks:
    def __init__(self):
        self.tasks = {
            "object_detection": "Detect objects in an image.",
            "image_classification": "Classify the content of an image.",
            "face_recognition": "Recognize faces in an image.",
            "image_segmentation": "Segment different parts of an image.",
            "optical_character_recognition": "Extract text from images."
        }

    
    def go_to(self, command: GoTo):
        return Status.EXECUTION_SUCCESS, "arrived to: " + command.location_to_go
    
    def pick_object(self, command: PickObject):
        return Status.EXECUTION_SUCCESS, "picked up: " + command.object_to_pick
    
    def place_object(self, command: PlaceObject):
        return Status.EXECUTION_SUCCESS, "placed object"
    
    def say_with_context(self, command: SayWithContext):
        return Status.EXECUTION_SUCCESS, "success"
    
    def answer_question(self, command: AnswerQuestion):
        return Status.EXECUTION_SUCCESS, "success"
    
    def get_visual_info(self, command: GetVisualInfo):
        return Status.EXECUTION_SUCCESS, "Fanta"
    
    def give_object(self, command: GiveObject):
        return Status.EXECUTION_SUCCESS, "object given"
    
    def follow_person_until(self, command: FollowPersonUntil):
        if command.destination == "cancelled":
            return Status.EXECUTION_SUCCESS, "followed user until cancelled"
        return Status.EXECUTION_SUCCESS, "arrived to " + command.destination
    
    def guide_person_to(self, command: GuidePersonTo):
        return Status.EXECUTION_SUCCESS, "arrived to " + command.destination_room

    def get_person_info(self, command: GetPersonInfo):
        if command.info_type == "gesture":
            return Status.EXECUTION_SUCCESS, "pointing to the right"
        elif command.info_type == "pose":
            return Status.EXECUTION_SUCCESS, "standing"
        
        return Status.EXECUTION_SUCCESS, "Name: John"
    
    def count(self, command: Count):
        return Status.EXECUTION_SUCCESS, "4"
    
    def find_person(self, command: FindPersonByName):
        return Status.EXECUTION_SUCCESS, "found " + command.attribute_value
    
    def find_person_by_name(self, command: FindPersonByName):
        return Status.EXECUTION_SUCCESS, f"found {command.name}"