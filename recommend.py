import streamlit as st



def recommend():   
    
    # title for project
    st.title("Product Recommendation") 
    
    # input to get from the user
    num1 = st.number_input("Enter the product")
    
    #num2 = st.number_input("Enter the second number")

    add_button = st.button("Submit")
#     subtract_button = st.button("Subtract")
#     multiply_button = st.button("Multiply")
#     divide_button = st.button("Divide")

#     if add_button:
#         result = num1 + num2
#         st.write(f"The sum of {num1} and {num2} is {result}")
#     elif subtract_button:
#         result = num1 - num2
#         st.write(f"The difference between {num1} and {num2} is {result}")
#     elif multiply_button:
#         result = num1 * num2
#         st.write(f"The product of {num1} and {num2} is {result}")
#     elif divide_button:
#         if num2 == 0:
#             st.write("Cannot divide by zero")
#         else:
#             result = num1 / num2
#             st.write(f"The quotient of {num1} and {num2} is {result}")

if __name__ == "__main__":
   recommend()
