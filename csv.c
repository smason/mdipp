#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>

#include <assert.h>

#include "csv.h"

#if defined(__linux__) || defined(WIN32)
/* stolen from http://geneslinuxbox.net:6309/gene/toolshed/libdecb/libdecbsrec.c */
static inline int digittoint(int c)
/* implemented based on OSX man page */
{
	/* if not 0-9, a-f, or A-F then return 0 */
	if (!isxdigit(c))
		return 0;

	if (isdigit(c))
		return c - '0';

	if (isupper(c))
		return c - 'A' + 10;

	/* not 0-9, not A-F, must be a-f */
	return c - 'a' + 10;
}
#endif

int
csv_parse(struct csvfile *csv, FILE *fd)
{
  csv->fd      = fd;
  csv->buf     = malloc( csv->buflen   = 1024);
  csv->fields  = malloc((csv->fieldlen = 128) * sizeof(char*));
  csv->nfields = 0;

  if (!csv->buf || !csv->fields) {
    if (csv->buf)    free(csv->buf);
    if (csv->fields) free(csv->fields);
    memset(csv, 0, sizeof(*csv));
    return -1;
  }

  return 0;
}

void
csv_close (struct csvfile *csv)
{
  assert(csv->fd);

  fclose(csv->fd);
  free(csv->buf);
  free(csv->fields);
  memset(csv, 0, sizeof(*csv));
}

static char *
csv_doublebuf (struct csvfile *csv, char *cur)
{
  size_t len = csv->buflen * 2;
  char *p = realloc (csv->buf, len);
  if (!p) {
    // it doesn't hurt to leave the old text buffer around, I expect
    // the user to still call csv_close() which will free it
    return 0;
  }
  // move our pointers across to point into the new buffer
  for (int i = 0; i < csv->nfields; i++) {
    csv->fields[i] = (csv->fields[i] - csv->buf) + p;
  }
  cur = (cur - csv->buf) + p;
  csv->buf    = p;
  csv->buflen = len;
  return cur;
}

static char *
csv_readquoted (struct csvfile *csv, char *buf, int quote, int *finalchr)
{
  FILE *fd = csv->fd;
  char *end = csv->buf + csv->buflen;
  int chr = getc_unlocked (fd);
  while (chr != EOF && chr != quote) {
    /* need space for the longest malformed escape and a potential
     * trailing @null char */
    if (buf + 5 >= end) {
      buf = csv_doublebuf (csv, buf);
      if (!buf)	return 0;
    }
    if (chr == '\\') {
      chr = getc_unlocked (fd);
      switch (chr) {
      case 'a':  chr = '\a'; break;
      case 'b':  chr = '\b'; break;
      case 'f':  chr = '\f'; break;
      case 'r':  chr = '\r'; break;
      case 'n':  chr = '\n'; break;
      case 't':  chr = '\t'; break;
      case '\\': chr = '\\'; break;
      case '\'': chr = '\''; break;
      case '"':  chr = '\"'; break;
      case 'x': {
	int c1 = getc_unlocked (fd);
	if (isxdigit(c1)) {
	  int c2 = getc_unlocked (fd);
	  if (isxdigit(c2)) {
	    chr = digittoint (c1) << 4 | digittoint (c2);
	  } else {
	    *(buf++) = '\\';
	    *(buf++) = 'x';
	    *(buf++) = c1;
	    chr = c2;
	  }
	} else {
	  *(buf++) = '\\';
	  *(buf++) = 'x';
	  chr = c1;
	}
	break;
      }
      default: {
	if (chr >= '0' && chr <= '7') {
	  int c2 = getc_unlocked (fd);
	  if (c2 >= '0' && c2 <= '7') {
	    int c3 = getc_unlocked (fd);
	    if (c3 >= '0' && c3 <= '7') {
	      chr = (((digittoint (chr) << 3) | 
		      (digittoint (c2)) << 3) |
		     digittoint (c3));
	    } else {
	      *(buf++) = '\\';
	      *(buf++) = chr;
	      *(buf++) = c2;
	      chr = c3;
	    }
	  } else {
	    *(buf++) = '\\';
	    *(buf++) = chr;
	    chr = c2;
	  }
	} else {
	  *(buf++) = '\\';
	}
      }
      }
    }
    *(buf++) = chr;
    chr = getc_unlocked (fd);
  }
  if (chr == quote)
    chr = getc_unlocked (fd);
  *finalchr = chr;
  return buf;
}

const char **
csv_readrow (struct csvfile *csv, size_t *nfields)
{
  FILE *fd = csv->fd;
  csv->nfields = 0;

  /** Lock the file so we can use unlocked @{getc()}s which are *much*
   * faster---need to make sure that I unlock before returning
   */
  flockfile (fd);

  int chr = getc_unlocked(fd);
  // starting at an end of file, this really is the end
  if (chr == EOF) {
    *nfields = -1;
    funlockfile (fd);
    return NULL;
  }

  csv->nfields = 0;

  char *buf = csv->buf, *end = csv->buf + csv->buflen;
  while (1) {
    if (csv->nfields+1 == csv->fieldlen) {
      size_t len = csv->fieldlen * 2;
      char **p = realloc (csv->fields, len*sizeof(char*));
      if (!p) {
	// again, can leave csv->fields around to be freed by
	// csv_close()
	funlockfile (fd);
	return 0;
      }
      csv->fields   = p;
      csv->fieldlen = len;
    }
    csv->fields[csv->nfields++] = buf;
    while (chr != EOF && chr != '\r' && chr != '\n' && chr != ',') {
      if (chr == '"' || chr == '\'') {
	buf = csv_readquoted (csv, buf, chr, &chr);
	if (!buf) {
	  funlockfile (fd);
	  return 0;
	}
	end = csv->buf + csv->buflen;
      } else {
	// enough space for the null char
	if (buf+2 >= end) {
	  buf = csv_doublebuf (csv, buf);
	  if (!buf) {
	    funlockfile (fd);
	    return 0;
	  }
	  end = csv->buf + csv->buflen;
	}
	*(buf++) = chr;
	chr = getc_unlocked (fd);
      }
    }
    *(buf++) = 0;
    if (chr != ',')
      break;
    chr = getc_unlocked (fd);
  }
  if (chr == '\r') {
    chr = getc_unlocked (fd);
    if (chr != '\n') {
      ungetc (chr, fd);
    }
  }

  funlockfile (fd);

  *nfields = csv->nfields;
  return (const char**)csv->fields;
}
