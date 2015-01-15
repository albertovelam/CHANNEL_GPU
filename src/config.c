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
  config_lookup_int(config, "application.NX", &(*domain).nx);
  config_lookup_int(config, "application.NY", &(*domain).ny);
  config_lookup_int(config, "application.NZ", &(*domain).nz);
  return;
}

