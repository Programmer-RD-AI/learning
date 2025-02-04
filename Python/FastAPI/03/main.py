import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import FastAPI app

client = TestClient(app)


# Test registration endpoint
def test_register_user():
    response = client.post(
        "/register/",
        json={
            "username": "john",
            "email": "john@example.com",
            "password": "securepassword",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "username": "john",
        "email": "john@example.com",
        "password": "securepassword",
    }


# Test item addition with background task
def test_add_item():
    response = client.post(
        "/add_item/",
        json={"name": "Item1", "description": "A sample item", "price": 29.99},
    )
    assert response.status_code == 200
    assert "id" in response.json()  # Ensure the response contains the ID
    assert response.json()["name"] == "Item1"


# Test file upload endpoint with valid file type
def test_upload_file_valid():
    with open("test_image.png", "wb") as f:
        f.write(b"fake image data")  # Create a dummy PNG file for testing

    with open("test_image.png", "rb") as f:
        response = client.post(
            "/uploadfile/", files={"file": ("test_image.png", f, "image/png")}
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "test_image.png"

    # Clean up the test file
    os.remove("test_image.png")


# Test file upload endpoint with invalid file type
def test_upload_file_invalid():
    with open("test_image.txt", "wb") as f:
        f.write(b"dummy text file data")  # Create a dummy TXT file

    with open("test_image.txt", "rb") as f:
        response = client.post(
            "/uploadfile/", files={"file": ("test_image.txt", f, "text/plain")}
        )

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Invalid file type. Only PNG and JPEG are allowed."
    )

    # Clean up the test file
    os.remove("test_image.txt")


# Test the token endpoint
def test_token():
    response = client.post(
        "/token", data={"username": "john", "password": "securepassword"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


# Test error handling endpoint
def test_error_handling():
    response = client.get("/error/")
    assert response.status_code == 404
    assert response.json()["detail"] == "This is a custom error!"
