import settings

def convert_text_to_html(full_html, current_text, current_lang):
    '''
        style text results in html format
    '''

    if current_lang == "he" or current_lang == "ar":
        current_line = f"<p style='text-align:right;'> {current_text} </p>"
    else:
        current_line = f"<p style='text-align:left;'> {current_text} </p>"

    full_html.append(current_line)
    return full_html

def build_html_table(all_results):

    new_line   = False
    html_table = []
    html_table.append("<table border='1'  align='center' style='font-size:24px'>")

    count_num_of_results = 0
    for i, (text, lang) in enumerate(reversed(all_results)):

        # -- first line -> start with text and not with new line
        if (i == 0) and (text == "\n"):
            continue

        # if we have enogth results
        if count_num_of_results >= settings.MAX_SAVED_RESULTS:
            continue

        # --- if need to add new line
        if (text == "\n"):
            if new_line is True:
                continue
            else:
                new_line             = True

        else:
            new_line = False

        # -- ad current text
        if lang == "he":
            html_table.append(f"<tr align='right'>")
        else:
            html_table.append(f"<tr align='left'>")
        html_table.append(f"<td> {settings.LANGUAGES[lang]} </td>")
        html_table.append(f"<td> {text}</td>")
        html_table.append("</tr>")

        count_num_of_results = count_num_of_results + 1

    html_table.append("</table>")

    return html_table