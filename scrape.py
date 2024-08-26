from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

def script(state, commodity, district):
    initial_url = "https://agmarknet.gov.in/SearchCmmMkt.aspx"

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-gpu")

    driver = None
    try:
        # Setup Chrome driver with the specified options
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(initial_url)

        print("Selecting Commodity")
        commodity_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'ddlCommodity'))
        )
        Select(commodity_dropdown).select_by_visible_text(commodity)

        print("Selecting State")
        state_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'ddlState'))
        )
        Select(state_dropdown).select_by_visible_text(state)

        print("Entering Date")
        today = datetime.now()
        desired_date = today - timedelta(days=15)
        date_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "txtDate"))
        )
        date_input.clear()
        date_input.send_keys(desired_date.strftime('%d-%b-%Y'))

        print("Clicking Go button")
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'btnGo'))
        )
        button.click()

        time.sleep(10)

        print("Selecting District")
        district_dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'ddlDistrict'))
        )
        Select(district_dropdown).select_by_visible_text(district)

        print("Clicking Go button again")
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'btnGo'))
        )
        button.click()

        time.sleep(5)

        print("Waiting for the table to load")
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'cphBody_GridPriceData'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        data_list = []
        for row in soup.find_all("tr"):
            data_list.append(row.text.replace("\n", "_").replace("  ", "").split("__"))

        jsonList = []
        for i in data_list[4:len(data_list) - 1]:
            d = {}
            d["S.No"] = i[1]
            d["City"] = i[2]
            d["Market"] = i[3]
            d["Commodity"] = i[4]
            d["Min Prize"] = i[7]
            d["Max Prize"] = i[8]
            d["Model Prize"] = i[9]
            d["Date"] = i[10]
            jsonList.append(d)

        print(jsonList)
        return jsonList

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if driver:
            driver.quit()

# Example call to the script function
#script('Maharashtra', 'Soyabean', 'Latur')
