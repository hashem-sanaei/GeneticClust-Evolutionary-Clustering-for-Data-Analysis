from selenium import webdriver
import time

# Read lessons from file
with open('lessons.txt', 'r') as file:
    lessons = [line.strip().split(',') for line in file]

# Start WebDriver
driver = webdriver.Chrome('/path/to/chromedriver')  # Update the path to chromedriver
driver.get("https://portal.aut.ac.ir/aportal/LoginRole.jsp")

# Assuming you've logged in manually or have code for it

# Navigate to the lesson selection page
# Use driver.find_element_by... to locate and interact with elements

for lesson_code, lesson_group in lessons:
    # Find and fill the lesson code input
    # Find and fill the lesson group input

    # Solve CAPTCHA here (manually or using a service)

    # Confirm the selection
    # driver.find_element_by... and click or interact as needed

    time.sleep(2)  # Adjust delay as necessary

# Close the driver after completion
driver.quit()
