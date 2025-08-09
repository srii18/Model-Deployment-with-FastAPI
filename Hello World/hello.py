# Import Union type to allow a variable to be one of multiple types (e.g., str or None)
from typing import Union

# Import FastAPI class to create the application
from fastapi import FastAPI

# Create a FastAPI instance (the core app object)
app = FastAPI()

# Define a route for the root URL ("/") using GET method
@app.get("/")
def read_root():
    # Return a simple JSON response
    return {"Hello": "World"}

# Define a route for items with a path parameter "item_id"
# Also accept an optional query parameter "q"
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    # Return both the path parameter and query parameter as JSON
    return {"item_id": item_id, "q": q}
