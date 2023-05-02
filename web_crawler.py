from tokenize import String
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
import csv




# connecting and getting contents of weg page

icon_label_list = []
tag_label_list = []
mp4_link_list = []


for i in range(46, 64):
    print(i)
    if i == 1:
        site= "https://www.flaticon.com/animated-icons-most-downloaded"
    else:
        site = "https://www.flaticon.com/animated-icons-most-downloaded/" + str(i)
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = Request(site,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')

    # find the mp4 files
    icon_divs = soup.find_all('li', {"class" : "animated--icon"})
    for icons in icon_divs:
        # grab mp4 link and link to icon's individual page
        mp4_link = icons['data-mp4']
        child_icon = icons.find_all('a')
        icon_url = child_icon[0]['href']
        icon_label = child_icon[0]['title']

        print(mp4_link)

        # local_mp4_file = "mp4/" + icon_label + ".mp4"
        local_mp4_file = 'mp4/' + '_'.join(mp4_link.split('/')[-3:])
        print(local_mp4_file)
        r = requests.get(mp4_link)  
        with open(local_mp4_file, 'wb') as f:
            for chunk in r.iter_content(10000):  # 10_000 bytes
                if chunk:
                    #print('.', end='')  # every dot will mean 10_000 bytes 
                    f.write(chunk) 
        
        # request access to individual icon link
        request_2 = Request(icon_url, headers = hdr)
        page = urlopen(request_2)
        soup_icon = BeautifulSoup(page, 'html.parser')

        tags = soup_icon.find_all('a', {"class" : "tag--related"})
        tag_list = ""
        for tag in tags:
            tag_list += tag.decode_contents()
            tag_list.strip()
        
        
        tag_temp = []
        temp = ""
        found = False
        for letter in tag_list:
            if letter.isalnum():
                temp += letter
                found = True
            else:
                if temp != '' and found:
                    tag_temp.append(temp)
                    temp = ""
                    found = False
        final_tag = ""
        for i  in range(len(tag_temp)):
            if (i == len(tag_temp) - 1):
                final_tag += tag_temp[i]
            else:
                final_tag += tag_temp[i]
                final_tag += ','

        icon_label_list.append(icon_label)
        tag_label_list.append(final_tag)
        mp4_link_list.append(local_mp4_file)
        

with open('icons.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(["Icon_name", "Icon_tags", "mp4_link"]) 
    for i in range(len(icon_label_list)):
        writer.writerow([icon_label_list[i], tag_label_list[i], mp4_link_list[i]])
        
    




    


    








