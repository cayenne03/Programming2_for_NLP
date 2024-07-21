import os
from lxml import etree


def split_xml_to_n_parts(input_xml: str, num_parts: int = 10, output_folder: str = 'xml_parts') -> None:
    tree = etree.parse(input_xml)
    root = tree.getroot()
    reviews = root.findall('Review')
    reviews_per_part = len(reviews) // num_parts

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_parts):
        part_root = etree.Element('Reviews')
        part_reviews = reviews[i * reviews_per_part:(i + 1) * reviews_per_part]

        for review in part_reviews:
            part_root.append(review)

        part_tree = etree.ElementTree(part_root)
        part_tree.write(os.path.join(output_folder, f"part{i + 1}.xml"), pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print(f"{num_parts} parts created in the folder '{output_folder}'.")


if __name__ == "__main__":
    split_xml_to_n_parts(input_xml="ABSA16_Restaurants_Train_SB1_v2.xml")

