import xml.etree.ElementTree as ET
import question_classifier_web as qc


import asyncio
import aiofiles
import aiohttp

base_url = 'http://qcapi.harishmadabushi.com'

HEADERS = {
    'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/45.0.2454.101 Safari/537.36'),
}

async def get_question_class(id, q_id, question):
    params = {"auth": "keho120l4l", "question": question}
    url = f'{base_url}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            print(resp)
            data = await resp.text()
    async with aiofiles.open(
            f'{str(id) + str(q_id) }.json', 'w') as file:
        print(data)
        await file.write(data)

def get_instances(file):
    tree = ET.parse(file)
    root = tree.getroot()

    questions = []

    def read_question(q):
        text = q.attrib['text']
        id = q.attrib['id']
        return id, text

    def read_instance(instance_object):
        id = instance_object.attrib['id']
        text = instance_object[0].text
        for question in instance_object[1]:
            q_id, text = read_question(question)
            questions.append((id, q_id, text))


    for child in root:
        read_instance(child)

    return questions

questions = get_instances('../../train-data.xml')

loop = asyncio.get_event_loop()
loop.run_until_complete(
    asyncio.gather(
        *(get_question_class(*question) for question in questions)
    )
)

