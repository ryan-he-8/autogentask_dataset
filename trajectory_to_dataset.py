from bs4 import BeautifulSoup
import re
import os
import jsonlines


class HTMLParser:

    def __init__(self, html_content):
        self.soup = BeautifulSoup(html_content, "html5lib")

    def get_intent_text(self):
        intent_text = ""
        table = self.soup.find("body")
        config_text= table.pre.contents[0]
        match = re.search(r"intent:\s*(.*)", config_text)
        if match:
            intent_text = match.group(1).strip()
        return intent_text
    
    def get_urls(self):
        urls = [h3.a["href"] for h3 in self.soup.find_all("h3", attrs={"class": "url"})]
        return urls

    def get_obs_trees(self):
        obs_trees = [ div.pre.contents[0] for div in self.soup.find_all("div", attrs={"class": "state_obv"})]
        return obs_trees

    def get_prev_actions(self):
        prev_actions = [ div.contents[0] for div in self.soup.find_all("div", attrs={'class': 'prev_action'})]
        return prev_actions
    
    def get_raw_predictions(self):
        raw_predictions = [ div.pre.contents[0] for div in self.soup.find_all("div", attrs={"class": "raw_parsed_prediction"})]
        return raw_predictions
    
def convert_extracted_data_to_conversation_format(intent_text: str, urls: list, obs_trees: list, prev_actions: list, raw_predictions: list) -> list:
    if len(urls) == len(obs_trees) == len(prev_actions) == len(raw_predictions):
        print("All lists are the same length.")
    else:
        print("Lists have different lengths.")
        exit()
    intro = "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.\n\nHere's the information you'll have:\nThe user's objective: This is the task you're trying to complete.\nThe current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\nThe current web page's URL: This is the page you're currently navigating.\nThe open tabs: These are the tabs you have open.\nThe previous action: This is the action you just performed. It may be helpful to track your progress.\n\nThe actions you can perform fall into several categories:\n\nPage Operation Actions:\n`click [id]`: This action clicks on an element with a specific id on the webpage.\n`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0.\n`hover [id]`: Hover over an element with id.\n`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n`scroll [direction=down|up]`: Scroll the page up or down.\n\nTab Management Actions:\n`new_tab`: Open a new, empty browser tab.\n`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.\n`close_tab`: Close the currently active tab.\n\nURL Navigation Actions:\n`goto [url]`: Navigate to a specific URL.\n`go_back`: Navigate to the previously viewed page.\n`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\nHomepage:\nIf you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\nhttp://homepage.com/password.html lists all the account name and password for the websites. You can use them to log in to the websites.\n\nTo be successful, it is very important to follow the following rules:\n1. You should only issue an action that is valid given the current observation\n2. You should only issue one action at a time.\n3. You should follow the examples to reason step by step and then issue the next action.\n4. Generate the action in the correct format. Start with a \"In summary, the next action I will perform is\" phrase, followed by action inside ``````. For example, \"In summary, the next action I will perform is ```click [1234]```\".\n5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop."
    results = []
    for i in range(len(urls)):
        template = "OBSERVATION:\n{observation}\nURL: {url}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}"
        current = template.format(
            objective=intent_text,
            url=urls[i],
            observation=obs_trees[i],
            previous_action=prev_actions[i],
        )
        raw_parsed_prediction = raw_predictions[i]

        messages = {"messages": [
            {"role": "system", "content": intro},
            {"role": "user", "content": current},
            {"role": "assistant", "content": raw_parsed_prediction}
        ]}
        results.append(messages)
    return results

def main():
    dataset = []
    for file in os.listdir(f"aggregated_trajectories"):
        with open(f"aggregated_trajectories/{file}", "r", encoding="utf-8") as f:
            html_content = f.read()
        htmlParser = HTMLParser(html_content=html_content)
        intent_text = htmlParser.get_intent_text()
        urls = htmlParser.get_urls()
        obs_trees = htmlParser.get_obs_trees()
        prev_actions = htmlParser.get_prev_actions()
        raw_predictions = htmlParser.get_raw_predictions()
        res = convert_extracted_data_to_conversation_format(
                    intent_text=intent_text,
                    urls=urls,
                    obs_trees=obs_trees,
                    prev_actions=prev_actions,
                    raw_predictions=raw_predictions
                )
        dataset = dataset + res

    with jsonlines.open("autogentask_dataset.jsonl", mode="w") as writer:
        writer.write_all(dataset)

if __name__ == "__main__":
    main()