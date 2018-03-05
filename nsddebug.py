#!/usr/bin/python

class nsd_debugger(object):

    def __init__(self, filename='/tmp/nsd-debug.html'):
        self.filename = filename

    def finish(self):
        self.fout.close()

    def start_new(self):
        self.fout = open(self.filename, 'w')

    def start_big_table(self, header, bgcolor='red', fgcolor='white', colspan=2  ):
        html = '<table width=100%%><tr><td align=center colspan=%d bgcolor=%s border=1><font color=%s><b>%s</b></font></td></tr>\n' % (colspan, bgcolor, fgcolor, header)
        self.fout.write(html)

    def finish_big_table(self):
        html = '</table>\n'
        self.fout.write(html)

    def write_row(self, elements):
        self.fout.write('<tr>\n')
        for e in elements:
            html = '<td width=%d%%>%s</td>\n' % (100 / len(elements), str(e))
            self.fout.write(html)
        self.fout.write('</tr>\n')

    def write_word_features(self, word, feature_dict):
        self.start_big_table('Word vector of %s' % word, colspan=2)
        for k in feature_dict:
            feature_list = feature_dict[k]
            for feature, val in feature_list:
                self.write_row( ['%s(<b>%s</b>)' % (k, feature), val] )
        self.finish_big_table()

