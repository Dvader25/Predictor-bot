# Copyright 2022 ChipChaddleson

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



# for the bot you will import weather adn then call weather.construct(location, goodTemp). save it to a var of type string
import requests
def construct(location, goodTemp, key):
    returnStatment = ""
    url = "https://weatherapi-com.p.rapidapi.com/forecast.json" # URL for the API


    
    querystring = {"q":{location},"days":"1"} # q = city, days = number of days

    headers = { 
        "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com",
        # CHANGE THIS SHIT IN PRODUCTION ISGT I WILL CRY 🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩
        "X-RapidAPI-Key": key # to get an api key go to https://rapidapi.com/weatherapi/api/weatherapi-com/ and create an acct, then press subscribe and choose the free option. 
    }

    response = requests.request("GET", url, headers=headers, params=querystring) # GET request

    timesOut = [] # list to store the times

    r = response.json() # convert the response to a json object

    for i in r['forecast']['forecastday'][0]["hour"]: # loop through the hours
        tempHour = i['temp_c'] # gets the temperature in celcius
        timeHour = i["time"].split(" ")[1] # gets the time in date formate 12/31/1234 12:00 and splits it at the " " and takes the 1nd index
        condHour = i['condition']['code'] # gets the condition, refer to https://www.weatherapi.com/docs/weather_conditions.json
        isDayHour = i['is_day'] # gets the day or night
        print(f"{timeHour}:") 
            
        if condHour <= 1009 and tempHour >= goodTemp and isDayHour != 0: # 1009 is the code for overcast, is day checks if its day LMAO, and i think anything above 5c is a good temp tbh
            print(f"{timeHour} is a good time to go outside, its {tempHour} degrees celcius and {condHour}") # just some logging

            timesOut.append( # add the time to the list 
                {
                    "time": timeHour,
                    "temp": tempHour,
                    "cond": i["condition"]["text"],
                    "out": True # out is true if its a good time to go outside
                }
            )
        else:
            # condition for if its a bad day to go outside
            print(f"bad weather {i['condition']['text']} and temprature is {tempHour}")
            timesOut.append(
                {
                    "time": timeHour,
                    "temp": tempHour,
                    "cond": i["condition"]["text"],
                    "out": False, # if its not a good time to go outside, it is false
                }
            )

        print("\n") # new line

    print(timesOut) # log the list


    print("\n\n\n ~~ times out ~~ \n\n") # 🎊🌟 fancy stuff 🎇✨
    for i in timesOut:
        # check for consecutive hours where [i]['out'] is true, and how many hours are true consecutively
        if i['out'] == True:
            count = 1 # this is the firt good hour
            for j in timesOut[timesOut.index(i)+1:]: # checks the next hour
                if j['out'] == True: # if the hour after is good
                    count += 1 
                else:
                    break # end the loop once if find a bad hour, execute next part

            print(f"{i['time']} to {timesOut[timesOut.index(i)+count]['time']} is a good time to go outside: \n") # prints a range of good times
            print("  " + "_" * 30)
            # print the time condition and temp for the hours between the two times
            for k in timesOut[timesOut.index(i):timesOut.index(i)+count]:
                print(f" | {k['time']} is {k['cond']} and {k['temp']} degrees celcius ") # print the time and condition
                
                returnStatment += (f" | {k['time']} is {k['cond']} and {k['temp']} degrees celcius \n") # this adds all the usefull data to return statment
            

            break
    return returnStatment


if __name__ == "__main__":
    construct("London", 5, "<REQUIRED>") # see line 27
    print("press ctrl + c to exit")
    while 1:
        pass # this will keep the program open untill your press ctrl + c