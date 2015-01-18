#include "channel.h"
#include <string.h>

config_t read_config_file(char* name){
  config_t config;
  config_init(&config);

  if ( !config_read_file(&config, name)){
    fprintf(stderr,  "%s:%d - %s\n", config_error_file(&config),
	    config_error_line(&config), config_error_text(&config));
    config_destroy(&config);
    exit(1);
  }
  return config;
}


void read_domain_from_config(domain_t *domain,config_t *config){
  double dummy;
  config_lookup_int(config, "application.NX", &(*domain).nx);
  config_lookup_int(config, "application.NY", &(*domain).ny);
  config_lookup_int(config, "application.NZ", &(*domain).nz);
  config_lookup_float(config, "application.LX", &dummy);
  domain->lx = PI * (float)dummy;
  config_lookup_float(config, "application.LZ", &dummy);
  domain->lz = PI * (float)dummy;
  config_lookup_float(config, "application.REYNOLDS", &dummy);
  domain->reynolds = (float)dummy;
  config_lookup_float(config, "application.MASSFLUX", &dummy);
  domain->massflux = (float)dummy;
  return;
}

void read_filenames_from_config(paths_t *path, config_t *config){
  const char *dummy;
  config_lookup_string(config, "application.input.G",&dummy);
  strcpy(path->ginput,dummy);
  config_lookup_string(config, "application.output.G",&dummy);
  strcpy(path->goutput,dummy);
  config_lookup_string(config, "application.input.DDV",&dummy);
  strcpy(path->ddvinput,dummy);
  config_lookup_string(config, "application.output.DDV",&dummy);
  strcpy(path->ddvoutput,dummy);
  config_lookup_string(config, "application.input.UMEAN",&dummy);
  strcpy(path->umeaninput,dummy);
  config_lookup_string(config, "application.output.UMEAN",&dummy);
  strcpy(path->umeanoutput,dummy);
  config_lookup_string(config, "application.PATH",&dummy);
  strcpy(path->path,dummy);
  config_lookup_int(config, "application.FREQ_STATS",&(*path).freq_stats);
  return;
}
