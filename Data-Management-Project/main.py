import json 

print("SHOPPING SIMULATOR")

# LOGIN MENU LOOP

def mainLogin():
  # LOAD USERS FROM TXT FILE
  users = loadUsers()
  while True:
    
    selection = getMenuSelection()

    # REGISTRATION
    if selection == "1":
      print("\nREGISTRATION")
      username = input("ENTER USERNAME: ")
      password = input("ENTER PASSWORD: ")  
      registerUser(users, username, password)

    # LOGIN
    elif selection == "2":
      print("\nLOGIN IN")
      username = input("USERNAME: ")
      password = input("PASSWORD: ")
      checkUser(users, username, password)

    # INVALID SELECTION  
    else:
      print("Invalid Selection")

# LOGIN MENU SELECTION
def getMenuSelection():
  print("\nUSER LOGIN SYSTEM")
  print("1. CREATE NEW USER")
  print("2. LOGIN")
  return input("Selection (1-2): ")

# LOAD USERS FROM TXT FILE
def loadUsers():
  file = open("users.txt", "r")
  data = json.load(file)
  file.close()
  return data

# SAVE USERS TO TXT FILE
def save(anArr):
  file = open("users.txt", "w")
  json.dump(anArr, file)
  file.close()

# CREATE NEW USER WITH PROPERTIES - USERNAME, PASSWORD, CART
def newUser(username, password):
  return {
		"username": username,
		"password": password,
		"cart": [ ]
	}

# REGISTRATION 
def registerUser(users, username, password):
  # SEARCH USERS TXT FILE TO FIND IF USERNAME IS TAKEN
  search = linearSearch(users, username, "username")

  # USERNAME IS NOT IN TXT FILE - CARRY ON WITH PROGRAM
  if search == -1:
    print("SUCCESSFULLY REGISTERED")
    users.append(newUser(username, password))
    save(users)
    mainMenu(users, username)
  # USERNAME IS TAKEN - LOOP THROUGH LOGIN MENU
  else: 
    print("USERNAME TAKEN")

# LINEAR SEARCH FUNCTION
def linearSearch(anArray, item, key):
  for el in range(0, len(anArray)):
    if (anArray[el][key] == item):
      return el
  return -1

# LOGIN USER CHECKER - CHECK IF USERNAME AND PASSWORD MATCH UP
def checkUser(users, username, password):
  # FIND INDEX OF USERNAME IN USERS TXT FILE
  Uindex = linearSearch(users, username, "username")
  # USERNAME INDEX NOT FOUND 
  if Uindex == -1:
    print("USERNAME NOT FOUND")
  else: 
    # USERNAME INDEX FOUND - CHECK IF PASSWORD MATCHES UP
    if users[Uindex]["password"] == password:
      # PASSWORD MATCHES UP
      print("\nWELCOME " + username)
      mainMenu(users, username)
    else:
      # PASSWORD DOESN'T MATCH UP
      print("PASSWORD INPUTTED INCORRECTLY")

# CREATE A PRODUCT WITH PROPERTIES - PRODUCT, DESCRIPTION, PRICE, CATEGORY
def newProduct(product, description, price, category):
	return {
	  "Product": product,
    "Description": description,
		"Price": price,
		"Category": category
  }

# ARRAY FOR PRODUCTS
products = [ ]
products.append(newProduct("carrots", "regular carrots", 5, "food"))
products.append(newProduct("chedder cheese", "regular chedder cheese", 8, "food"))
products.append(newProduct("television", "40in. flat screen television", 400, "technology"))
products.append(newProduct("chair", "small wooden chair", 25, "furniture"))

# MAIN MENU FOR SIMULATOR
def mainMenu(users, username):
  while True: 
    selection = getStoreMenuSelection()

    # DISPLAY ALL PRODUCTS
    if selection == "1":
      displayAllProducts()
    # FILTER PRODUCTS BY CATEGORIES
    elif selection == "2":
      category = input("Enter a category to search for: ").lower()
      filterCategories(category)
    # SORT PRODUCTS BY PRICE
    elif selection == "3":
      bubbleSort(products)
    # ADD PRODUCT TO CART
    elif selection == "4":
      addtoCart(users, username)
    # REMOVE PRODUCT FROM CART
    elif selection == "5":
      removefromCart(users, username)
    # SHOW PRODUCTS IN CART
    elif selection == "6":
      showCart(users, username)
    # EXIT SIMULATOR AND GO BACK TO LOGIN SYSTEM
    elif selection == "7":
      print("EXIT")
      return False
    # USER DID NOT INPUT A NUMBER (1-7)
    else:
      print("Invalid Selection")

# STORE MENU SELECTION
def getStoreMenuSelection():
  print("\nMENU")
  print("1. DISPLAY ALL PRODUCTS")
  print("2: FILTER PRODUCTS")
  print("3: SORT PRODUCTS")
  print("4. ADD TO CART")
  print("5. REMOVE FROM CART")
  print("6. SHOW CART")
  print("7. EXIT")
  return input("Selection (1-7): ")

# DISPLAY ALL PRODUCTS
def displayAllProducts():
  for product in products:
    print(f"Product: {product['Product']}" + " " + f"Description:{product['Description']}" + " " + f"Price:{product['Price']}"+ " " + f"Category:{product['Category']}")

# FILTER PRODUCTS BY CATEGORIES 
def filterCategories(category):
  for product in products:
    # IF CATEGORY EXISTS PRINT ALL PRODUCTS IN THE CATEGORY
    if category in product['Category']:
      print(f"Product: {product['Product']}" + " " + f"Description:{product['Description']}" + " " + f"Price:{product['Price']}"+ " " + f"Category:{product['Category']}")

# BUBBLE SORT FOR PRICE
def bubbleSort(anArray): 
  for i in range(0, len(anArray) - 1):
    for el in range(0, len(anArray) - i - 1):
      if anArray[el]['Price'] > anArray[el + 1]['Price']:
        anArray[el], anArray[el + 1] = anArray[el + 1], anArray[el]

# ADD TO CART
def addtoCart(users, username):
  addCart = input("ADD SOMETHING TO CART: ").lower()
  # FIND INDEX OF ITEM IN PRODUCTS ARRAY
  productIndex = linearSearch(products, addCart, "Product")

  # INDEX = -1 - PRODUCT DOES NOT EXIST
  if productIndex == -1:
    print("PRODUCT NOT FOUND")
  # PRODUCT EXISTS
  else:
    for el in range(0, len(users)):
      # ADD PRODUCT TO CART OF THE USER AND SAVE
      if (users[el]["username"] == username):
        # IF PRODUCT NOT IN USERS CART ALREADY
        if addCart not in users[el]["cart"]:
          users[el]["cart"].append(addCart)
          print("PRODUCT ADDED TO CART")
          save(users)
        # IF PRODUCT IN USERS CART ALREADY
        else:
          print("PRODUCT ALREADY IN CART")

# REMOVE FROM CART
def removefromCart(users, username):
  removeCart = input("REMOVE SOMETHING FROM CART: ").lower()
  # FIND INDEX OF ITEM IN PRODUCTS ARRAY
  productIndex = linearSearch(products, removeCart, "Product")

  # INDEX = -1 - PRODUCT DOES NOT EXIST
  if productIndex == -1:
    print("PRODUCT NOT FOUND")
  # PRODUCT EXISTS
  else:
    for el in range(0, len(users)):
      if (users[el]["username"] == username):
        # REMOVE PRODUCT FROM CART OF THE USER AND SAVE
        if removeCart in users[el]["cart"]:  
          users[el]["cart"].remove(removeCart)
          print("PRODUCT REMOVED FROM CART")
          save(users)
        # PRODUCT EXISTS BUT IS NOT IN THE USERS CART
        else:
          print("PRODUCT NOT IN CART")

# SHOW USERS CART
def showCart(users, username):
  print("\nCART: ")  
  for el in range(0, len(users)):
    if (users[el]["username"] == username):
      print(*users[el]["cart"], sep = ", ")    

mainLogin()