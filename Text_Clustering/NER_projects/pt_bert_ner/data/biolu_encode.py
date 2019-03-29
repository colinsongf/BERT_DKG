import argparse

parser = argparse.ArgumentParser(description='Change encoding from BIO to BIOLU')
parser.add_argument('input', metavar='-i', type=str, help='The path to the original file with BIO encoding')
parser.add_argument('output', metavar='-o', type=str, help='The name of your BIOLU encoded file')
args = parser.parse_args()

input_file = args.input
output_file = args.output


def read_file(input_file):
    with open(input_file, 'rb') as f:
        return f.read().decode('ASCII').split('\n')


def write_line(new_label: str, prev_label: str, line_content: list, output_file):
    new_iob = new_label + prev_label
    line_content[3] = new_iob
    current_line = ' '.join(line_content)
    output_file.write(current_line + '\n')

def not_same_tag(tag1, tag2):
    return tag1.split("-")[-1]!=tag2.split("-")[-1]

def convert(input_file, output_path):
    output_file = open(output_path, 'w')

    for i in range(len(input_file) + 1):

        try:
            current_line = input_file[i]

            if '-DOCSTART-' in current_line:
                output_file.write(current_line)
            elif len(current_line) == 1:
                output_file.write(current_line)

            else:
                prev_iob = ""
                next_iob = ""
                prev_line = None
                next_line = None

                try:
                    prev_line = input_file[i - 1]
                    next_line = input_file[i + 1]

                    if len(prev_line.strip()) > 0:
                        prev_line_content = prev_line.split()
                        prev_iob = prev_line_content[-1]

                    if len(next_line.strip()) > 0:
                        next_line_content = next_line.split()
                        next_iob = next_line_content[-1]

                except IndexError:
                    pass

                current_line_content = current_line.strip().split()
                current_iob = current_line_content[-1]

                # Outside entities
                if current_iob == 'O':
                    output_file.write(current_line)

                # Unit length entities
                elif current_iob.startswith("B-") and \
                        (next_iob == 'O' or len(next_line.strip()) == 0 or next_iob.startswith("B-")):
                    write_line('S-', current_iob[2:], current_line_content, output_file)

                # First element of chunk
                elif current_iob.startswith("B-") and \
                        (not not_same_tag(current_iob,next_iob) and next_iob.startswith("I-")):
                    write_line('B-', current_iob[2:], current_line_content, output_file)

                # Last element of chunk
                elif current_iob.startswith("I-") and \
                        (next_iob == 'O' or len(next_line.strip()) == 0 or next_iob.startswith("B-")):
                    write_line('E-', current_iob[2:], current_line_content, output_file)

                # Inside a chunk
                elif current_iob.startswith("I-") and \
                        next_iob.startswith("I-"):
                    write_line('I-', current_iob[2:], current_line_content, output_file)

        except IndexError:
            pass


bio = read_file(input_file)
convert(bio, output_file)
