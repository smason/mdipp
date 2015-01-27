struct csvfile {
  FILE *fd;
  char *buf, **fields;
  size_t buflen, fieldlen, nfields;
};

int  csv_parse(struct csvfile *csv, FILE *fd);
void csv_close (struct csvfile *csv);
const char ** csv_readrow (struct csvfile *csv, size_t *nfields);
