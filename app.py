import psycopg2
from psycopg2 import OperationalError
import argparse
from datetime import datetime

def connect_db():
    # Attempt to establish a connection to the database.
    try:
        conn = psycopg2.connect(
            dbname="", # Insert your db name
            user="postgres", # Fefault is postgres, but yours may be different
            password="", # Insert your password
            host="localhost",
            port=5432
        )
        return conn  # Return the connection object if successful.
    except OperationalError as e:
        print(f"An error occurred: {e}")
        return None  # Return None if the connection attempt fails.

def getAllStudents():
    # Retrieve and display all student records from the database.
    conn = connect_db()  # Establish a database connection.
    cur = conn.cursor()  # Create a cursor object to execute SQL commands.
    cur.execute("SELECT * FROM students;")  # SQL command to select all records.
    records = cur.fetchall()  # Fetch all rows of the query result.
    print("Student Records:")
    for record in records:
        # Unpack each record for readability and format the enrollment date.
        student_id, first_name, last_name, email, enrollment_date = record
        formatted_date = enrollment_date.strftime("%Y-%m-%d")  # Format date.
        print(student_id, first_name, last_name, email, formatted_date)
    cur.close()
    conn.close()

def addStudent(first_name, last_name, email, enrollment_date):
    # Insert a new student record into the database.
    conn = connect_db()
    cur = conn.cursor()
    # SQL command to insert a new record with provided details.
    cur.execute("INSERT INTO students (first_name, last_name, email, enrollment_date) VALUES (%s, %s, %s, %s);",
                (first_name, last_name, email, enrollment_date))
    conn.commit()  # Commit the transaction to the database.
    print("Student added successfully.")
    cur.close()
    conn.close()

def updateStudentEmail(student_id, new_email):
    # Update the email address of a specific student.
    conn = connect_db()
    cur = conn.cursor()
    # SQL command to update the email of a student by student_id.
    cur.execute("UPDATE students SET email = %s WHERE student_id = %s;", (new_email, student_id))
    conn.commit()  # Commit the changes to the database.
    print("Student email updated successfully.")
    cur.close()
    conn.close()

def deleteStudent(student_id):
    # Delete a specific student record from the database.
    conn = connect_db()
    cur = conn.cursor()
    # SQL command to delete a record by student_id.
    cur.execute("DELETE FROM students WHERE student_id = %s;", (student_id,))
    conn.commit()  # Commit the transaction to ensure changes are saved.
    print("Student deleted successfully.")
    cur.close()
    conn.close()

def main_menu():
    # Main menu for user interaction with the database management system.
    print("Welcome to the Student Database Management System")
    print("1. View all students")
    print("2. Add a new student")
    print("3. Update a student's email")
    print("4. Delete a student")
    print("5. Exit")

    while True:
        choice = input("Enter your choice: ")  # Prompt user for their choice.

        if choice == "1":
            getAllStudents()  # Display all student records.
        elif choice == "2":
            # Prompt for new student details and add the student.
            first_name = input("Enter first name: ")
            last_name = input("Enter last name: ")
            email = input("Enter email: ")
            enrollment_date = input("Enter enrollment date (YYYY-MM-DD): ")
            addStudent(first_name, last_name, email, enrollment_date)
        elif choice == "3":
            # Prompt for student ID and new email, then update the email.
            student_id = int(input("Enter student ID: "))
            new_email = input("Enter new email: ")
            updateStudentEmail(student_id, new_email)
        elif choice == "4":
            # Prompt for student ID to delete, then delete the student.
            student_id = int(input("Enter student ID to delete: "))
            deleteStudent(student_id)
        elif choice == "5":
            print("Exiting the program.")
            break  # Exit the loop to end the program.
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main_menu()
