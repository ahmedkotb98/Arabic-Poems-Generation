import scrapy
from bs4 import BeautifulSoup


class AldiwanSpider(scrapy.Spider):
    name = 'aldiwan'
    allowed_domains = ['www.aldiwan.net']
    start_urls = ['http://www.aldiwan.net/']

    def start_requests(self):
        for poem_id in range(24, 71232):
            reqeust = scrapy.Request(url="https://www.aldiwan.net/poem" + str(poem_id) + ".html",
                                     callback=self.parse)
            reqeust.meta['poem_id'] = poem_id
            yield reqeust

    def parse(self, response):
        poem_id = response.meta['poem_id']
        poem_link = "https://www.aldiwan.net/poem" + str(poem_id) + ".html"
        info = get_poem_info(response.text)
        if info == "error":
            print('Poems: not found {}'.format(poem_id))
            return
        poem_text, poem_info, poem_name, cat_name, poet_name = info

        poem_topics = ""
        poem_type = ""
        poem_meter = ""
        first_line = ""
        rhyme = ""
        for item in poem_info:
            name, cat = item
            if "Topics" in cat:
                poem_topics = name
            if "Type" in cat:
                poem_type = name
            if "sea" in cat:
                poem_meter = name
                
        if 'عموديه' in poem_type or poem_meter or poem_type == "":
            splited_text = poem_text.split('\n')
            black_list = ['***', '* * *']
            splited_text = [
                t
                for t in splited_text
                if t.strip() not in black_list
            ]
            num_of_lines = len(splited_text)
            check = num_of_lines % 2
            if check == 0:
                poem_text = [[splited_text[i], splited_text[i + 1]] for i in range(0, len(splited_text), 2)]
                first_line = poem_text[0]
                rhyme = first_line[1].split()[-1]
            else:
                poem_text = [[splited_text[i], splited_text[i + 1]] for i in range(0, num_of_lines - 1, 2)]
                poem_text.append([splited_text[-1]])
                first_line = poem_text[0]
                rhyme = first_line[0].split()[-1]
        else:
            splited_text = poem_text.split('\n')
            black_list = ['***', '* * *']
            splited_text = [
                t+'\n'
                for t in splited_text
                if t.strip() not in black_list
            ]
            num_of_lines = len(splited_text)
            poem_text = splited_text

        # update csv file
        row_dict = {'poem_id': poem_id, 'era-and-country_ar': cat_name,
                    'poet_name': poet_name, 'poem_name': poem_name.strip(),
                    'category_ar': poem_topics, 'type_ar': poem_type,
                    'meter': poem_meter, 'number_of_lines': num_of_lines,
                    'first_line': first_line, 'rhyme': rhyme, 'poem_text': poem_text,
                    'poem_link': poem_link}
        yield row_dict


def get_poem_info(data):
    # use soup html parser
    soup = BeautifulSoup(data, "html.parser")
    try:
        # get all poets names/links within this country/era
        poem_tags = soup.find('div', {'class': 'bet-1'}).find_all('h3')
        poem_tags += soup.find('div', {'class': 'bet-1'}).find_all('h4')
        poem_text = ""
        for tag in poem_tags:
            text = tag.get_text(strip=True, separator="\n") + '\n'
            if len(text):
                poem_text += text
    except:
        return "error"

    # info
    try:
        peom_info_tags = soup.find('div', {'class': 'tips row content'}).find_all('div', {'class': 'col'})
        poem_info = [
            (tag.findChild().text, tag.findChild().get('href').split('.html')[0])
            for tag in peom_info_tags
            if len(tag.findChild().text.strip())
        ]
    except:
        poem_info = ""

    try:

        # poem_name, cat, poet_name
        title_bar = soup.find('div', {'class': 'col-12 relative'})
        poem_name = title_bar.find('h1', {'class': 'h2'}).text
        info = title_bar.find('h2').find_all('a')
        cat_name = info[1].text
        poet_name = info[2].text
    except:
        poem_name = ""
        cat_name = ""
        poet_name = ""

    return poem_text.strip(), poem_info, poem_name, cat_name, poet_name
