import json
from datetime import datetime
from typing import Union
import categorization
from types import SimpleNamespace

from utils.baml_client.types import (
    AnswerQuestion,
    CommandListLLM,
    Count,
    FindPerson,
    FindPersonByName,
    FollowPersonUntil,
    GetPersonInfo,
    GetVisualInfo,
    GiveObject,
    GoTo,
    GuidePersonTo,
    PickObject,
    PlaceObject,
    SayWithContext,
)

InterpreterAvailableCommands = Union[
    CommandListLLM,
    GoTo,
    PickObject,
    FindPersonByName,
    FindPerson,
    Count,
    GetPersonInfo,
    GetVisualInfo,
    AnswerQuestion,
    FollowPersonUntil,
    GuidePersonTo,
    GiveObject,
    PlaceObject,
    SayWithContext,
]



# /////////////////embeddings services/////
def add_command_history(self, command: InterpreterAvailableCommands, result, status):
    collection = "command_history"

    document = [command.action]
    metadata = [
        {
            "command": str(command),
            "result": result,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
    ]

    request = SimpleNamespace(
        document=document,
        metadata=metadata = json.dumps(metadata),
        collection=collection
    )
    response = self.add_item_client(request)

    return response
def _add_to_collection(self, document: list, metadata: str, collection: str) -> str:
    request = SimpleNamespace(
        document=document,
        metadata=metadata = metadata,
        collection=collection
    )



def add_item(self, document: list, metadata: str) -> list[str]:
    return self._add_to_collection(document, metadata, "items")

def add_location(self, document: list, metadata: str) -> list[str]:
    return self._add_to_collection(document, metadata, "locations")

def query_item(self, query: str, top_k: int = 1) -> list[str]:
    return self._query_(query, "items", top_k)

def query_location(self, query: str, top_k: int = 1) -> list[str]:
    return self._query_(query, "locations", top_k)

# def find_closest(self, documents: list, query: str, top_k: int = 1) -> list[str]:
#     """
#     Method to find the closest item to the query.
#     Args:
#         documents: the documents to search among
#         query: the query to search for
#     Returns:
#         Status: the status of the execution
#         list[str]: the results of the query
#     """
#     self._add_to_collection(document=documents, metadata="", collection="closest_items")
#     Results = self._query_(query, "closest_items", top_k)
#     Results = self.get_name(Results)
#     print(f"find_closest result({query}): {str(Results)}")
#     return Status.EXECUTION_SUCCESS, Results

# def find_closest_raw(self, documents: str, query: str, top_k: int = 1) -> list[str]:
#     """
#     Method to find the closest item to the query.
#     Args:
#         documents: the documents to search among
#         query: the query to search for
#     Returns:
#         Status: the status of the execution
#         list[str]: the results of the query
#     """
#     self._add_to_collection(document=documents, metadata="", collection="closest_items")
#     Results = self._query_(query, "closest_items", top_k)
#     Logger.info(self.node, f"find_closest result({query}): {str(Results)}")
#     return Results

# @service_check("llm_wrapper_service", (Status.SERVICE_CHECK, ""), TIMEOUT)
# def answer_with_context(self, question: str, context: str) -> str:
#     """
#     Method to answer a question with context.
#     Args:
#         question: the question to answer
#         context: the context to use
#     Returns:
#         Status: the status of the execution
#         str: the answer to the question
#     """
#     self.node.get_logger().info(f"answer_with_context called with: {question}, {context}")

#     request = LLMWrapper.Request(question=question, context=context)
#     future = self.llm_wrapper_service.call_async(request)
#     rclpy.spin_until_future_complete(self.node, future)
#     return Status.EXECUTION_SUCCESS, future.result().answer

# def query_command_history(self, query: str, top_k: int = 1):
#     """
#     Method to query the command history collection.
#     Args:
#         query: the query to search for
#     Returns:
#         Status: the status of the execution
#         list[str]: the results of the query
#     """
#     return self._query_(query, "command_history", top_k)

# # /////////////////helpers/////
# def _query_(self, query: str, collection: str, top_k: int = 1) -> tuple[Status, list[str]]:
#     # Wrap the query in a list so that the field receives a sequence of strings.
#     request = QueryEntry.Request(query=[query], collection=collection, topk=top_k)
#     future = self.query_item_client.call_async(request)
#     rclpy.spin_until_future_complete(self.node, future)
#     if collection == "command_history":
#         self.node.get_logger().info(f"Querying command history: {future.result().results}")
#         results_loaded = json.loads(future.result().results[0])
#         sorted_results = sorted(
#             results_loaded["results"], key=lambda x: x["metadata"]["timestamp"], reverse=True
#         )
#         results_list = sorted_results[:top_k]
#     else:
#         results = future.result().results

#         results_loaded = json.loads(results[0])
#         results_list = results_loaded["results"]
#     return Status.EXECUTION_SUCCESS, results_list


# def get_context(self, query_result):
#     return self.get_metadata_key(query_result, "context")

# def get_command(self, query_result):
#     return self.get_metadata_key(query_result, "command")

# def get_result(self, query_result):
#     return self.get_metadata_key(query_result, "result")

# def get_status(self, query_result):
#     return self.get_metadata_key(query_result, "status")

# def get_name(self, query_result):
#     return self.get_metadata_key(query_result, "original_name")

# def categorize_objects(
#     self, table_objects: list[str], shelves: dict[int, list[str]]
# ) -> tuple[Status, dict[int, list[str]], dict[int, list[str]]]:
#     """
#     Categorize objects based on their shelf levels.

#     Args:
#         table_objects (list[str]): List of objects on the table.
#         shelves (dict[int, list[str]]): Dictionary mapping shelf levels to object names.

#     Returns:
#         dict[int, list[str]]: Dictionary mapping shelf levels to categorized objects.
#     """
#     Logger.info(self.node, "Sending request to categorize_objects")

#     try:
#         categories = self.get_shelves_categories(shelves)[1]
#         results = self.categorize_objects_with_embeddings(categories, table_objects)

#         objects_to_add = {key: value["objects_to_add"] for key, value in results.items()}
#         Logger.error(self.node, f"categories {categories}")

#         if "empty" in categories.values():
#             # add objects to add in shelves
#             for k, v in objects_to_add.items():
#                 for i in v:
#                     shelves[k].append(i)
#             categories = self.get_shelves_categories(shelves)[1]

#         Logger.error(self.node, f"THIS IS THE CATEGORIZED SHELVES: {categories}")
#         categorized_shelves = {
#             key: value["classification_tag"] for key, value in results.items()
#         }
#         for k, v in categorized_shelves.items():
#             if v == "empty":
#                 categorized_shelves[k] = categories[k]
#     #             categorized_shelves = {
#     #                 key: value["classification_tag"] for key, value in results.items()
#     #             }

#     except Exception as e:
#         self.node.get_logger().error(f"Error: {e}")
#         return Status.EXECUTION_ERROR, {}, {}

#     Logger.info(self.node, "Finished executing categorize_objects")

#     return Status.EXECUTION_SUCCESS, categorized_shelves, objects_to_add

# def get_shelves_categories(
#     self, shelves: dict[int, list[str]]
# ) -> tuple[Status, dict[int, str]]:
#     """
#     Categorize objects based on their shelf levels.

#     Args:
#         shelves (dict[int, list[str]]): Dictionary mapping shelf levels to object names.

#     Returns:
#         dict[int, str]: Dictionary mapping shelf levels to its category.
#     """
#     Logger.info(self.node, "Sending request to categorize_objects")

#     try:
#         request = CategorizeShelves.Request(shelves=String(data=str(shelves)), table_objects=[])

#         future = self.categorize_service.call_async(request)
#         Logger.info(self.node, "generated request")
#         rclpy.spin_until_future_complete(self.node, future, timeout_sec=25)
#         res = future.result()
#         Logger.info(self.node, "request finished")
#         Logger.info(self.node, "categorize_objects result: " + str(res))
#         # if res.status != Status.EXECUTION_SUCCESS:
#         #     Logger.error(self.node, f"Error in categorize_objects: {res.status}")
#         #     return Status.EXECUTION_ERROR, {}, {}

#         categorized_shelves = res.categorized_shelves
#         categorized_shelves = {k: v for k, v in enumerate(categorized_shelves)}
#     except Exception as e:
#         self.node.get_logger().error(f"Error: {e}")
#         return Status.EXECUTION_ERROR, {}

#     Logger.info(self.node, "get_shelves_categories:" + str(categorized_shelves))

#     return Status.EXECUTION_SUCCESS, categorized_shelves

# def get_subarea(self, query_result):
#     return self.get_metadata_key(query_result, "subarea")

# def get_area(self, query_result):
#     return self.get_metadata_key(query_result, "area")

# def get_metadata_key(self, query_result, field: str):
#     """
#     Extracts the field from the metadata of a query result.

#     Args:
#         query_result (tuple): The query result tuple (status, list of JSON strings)

#     Returns:
#         list: The 'context' field from metadata, or empty string if not found
#     """
#     try:
#         key_list = []
#         query_result = query_result[1]
#         for result in query_result:
#             metadata = result["metadata"]
#             if isinstance(metadata, list) and metadata:
#                 metadata = metadata[0]
#             result_key = metadata.get(field, "")  # safely get 'field'
#             key_list.append(result_key)
#         return key_list
#     except (IndexError, KeyError, json.JSONDecodeError) as e:
#         self.node.get_logger().error(f"Failed to extract context: {str(e)}")
#         return ""
# def categorize_object(self, categories: dict, obj: str):
#     """Method to categorize a list of objects in an array of objects depending on similarity"""

#     try:
#         category_list = []
#         categories_aux = categories.copy()
#         self.node.get_logger().info(f"OBJECT TO CATEGORIZE: {obj}")
#         for key in list(categories_aux.keys()):
#             if categories_aux[key] == "empty":
#                 self.node.get_logger().info("THERE IS AN EMPTY SHELVE")
#                 del categories_aux[key]

#         for category in categories_aux.values():
#             category_list.append(category)

#         results = self.find_closest_raw(category_list, obj)
#         results_distances = results[1][0]["distance"]

#         result_category = results[1][0]["document"]
#         self.node.get_logger().info(f"CATEGORY PREDICTED BEFORE THRESHOLD: {result_category}")
#         if "empty" in categories.values() and results_distances[0] > 1:
#             result_category = "empty"

#         self.node.get_logger().info(f"CATEGORY PREDICTED: {result_category}")

#         key_resulted = 2
#         for key in list(categories.keys()):
#             if str(categories[key]) == str(result_category):
#                 key_resulted = key
#             else:
#                 self.node.get_logger().info(
#                     "THE CATEGORY PREDICTED IS NOT CONTAINED IN THE REQUEST, RETURNING SHELVE 2"
#                 )

#         return key_resulted

#     except Exception as e:
#         self.node.get_logger().error(f"FAILED TO CATEGORIZE: {obj} with error: {e}")

# def categorize_objects_with_embeddings(self, categories: dict, obj_list: list):
#     """
#     Categorize objects based on their embeddings."""
#     self.node.get_logger().info(f"THIS IS THE CATEGORIES dict RECEIVED: {categories}")
#     self.node.get_logger().info(f"THIS IS THE obj_list LIST RECEIVED: {obj_list}")
#     results = {
#         key: {"classification_tag": categories[key], "objects_to_add": []}
#         for key in categories.keys()
#     }
#     for obj in obj_list:
#         index = self.categorize_object(categories, obj)
#         results[index]["objects_to_add"].append(obj)

#     self.node.get_logger().info(f"THIS IS THE RESULTS OF THE CATEGORIZATION: {results}")
#     return results
