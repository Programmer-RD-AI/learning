from graphviz import Digraph

def create_base_diagram(name, filename):
    diagram = Digraph(name, filename=filename)
    diagram.attr(
        rankdir='TB',
        splines='ortho',
        nodesep='0.5',
        ranksep='0.7',
        fontname='Arial',
        dpi='300'
    )
    
    diagram.attr('node',
        shape='record',
        style='filled',
        fillcolor='white',
        fontname='Arial',
        fontsize='10'
    )
    
    diagram.attr('edge',
        fontname='Arial',
        fontsize='8'
    )
    
    return diagram

def create_presentation_layer():
    diagram = create_base_diagram('Presentation_Layer', 'presentation_layer.png')
    
    # Define presentation layer classes with simplified labels
    diagram.node('AppointmentView', '''<f0> AppointmentView|<f1> - scanner: Scanner\l- appointmentController: AppointmentController\l|<f2> + showAppointmentMenu()\l+ displayMenu()\l+ getUserChoice()\l''')
    
    diagram.node('PatientView', '''<f0> PatientView|<f1> |<f2> + getPatientDetails()\l''')
    
    diagram.node('DermatologistView', '''<f0> DermatologistView|<f1> - scanner: Scanner\l|<f2> + selectDermatologist()\l+ addTimeSlotForDermatologist()\l''')
    
    # Add relationships
    diagram.edge('AppointmentView', 'PatientView')
    diagram.edge('AppointmentView', 'DermatologistView')
    
    return diagram

def create_service_layer():
    diagram = create_base_diagram('Service_Layer', 'service_layer.png')
    
    diagram.node('AppointmentService', '''<f0> AppointmentService|<f1> - appointmentRepository\l|<f2> + createAppointment()\l+ listAppointments()\l+ updateAppointment()\l''')
    
    diagram.node('DermatologistService', '''<f0> DermatologistService|<f1> - dermatologistRepository\l|<f2> + selectDermatologist()\l+ addTimeSlotForDermatologist()\l''')
    
    diagram.node('PatientService', '''<f0> PatientService|<f1> - patientRepository\l|<f2> + getPatientDetails()\l+ createPatient()\l''')
    
    return diagram

def create_repository_layer():
    diagram = create_base_diagram('Repository_Layer', 'repository_layer.png')
    
    diagram.node('AppointmentRepository', '''<f0> AppointmentRepository|<f1> - appointments: List\l|<f2> + save()\l+ findAll()\l+ findById()\l''')
    
    diagram.node('DermatologistRepository', '''<f0> DermatologistRepository|<f1> - dermatologists: List\l|<f2> + save()\l+ findAll()\l+ findById()\l''')
    
    diagram.node('PatientRepository', '''<f0> PatientRepository|<f1> - patients: List\l|<f2> + save()\l+ findAll()\l+ findById()\l''')
    
    return diagram

def create_domain_layer():
    diagram = create_base_diagram('Domain_Layer', 'domain_layer.png')
    
    diagram.node('Appointment', '''<f0> Appointment|<f1> - appointmentId: int\l- patient: Patient\l- dermatologist: DermatologistModel\l|<f2> + setStatus()\l+ getInvoice()\l+ getId()\l''')
    
    diagram.node('Patient', '''<f0> Patient|<f1> - patientId: int\l- name: String\l- age: int\l|<f2> + getId()\l+ getName()\l+ getAge()\l''')
    
    diagram.node('DermatologistModel', '''<f0> DermatologistModel|<f1> - dermatologistId: int\l- name: String\l|<f2> + checkAvailability()\l+ addTimeSlot()\l''')
    
    # Add relationships
    diagram.edge('Appointment', 'Patient', 'contains')
    diagram.edge('Appointment', 'DermatologistModel', 'contains')
    
    return diagram

def create_combined_diagram():
    diagram = create_base_diagram('Combined_System', 'combined_system.png')
    
    with diagram.subgraph(name='cluster_0') as presentation:
        presentation.attr(label='Presentation Layer', style='filled', fillcolor='#E8F4F9')
        presentation.node('AppointmentView', '''<f0> AppointmentView|<f1> |<f2> + showAppointmentMenu()\l''')
        presentation.node('PatientView', '''<f0> PatientView|<f1> |<f2> + getPatientDetails()\l''')
    
    with diagram.subgraph(name='cluster_1') as service:
        service.attr(label='Service Layer', style='filled', fillcolor='#E8F6F3')
        service.node('AppointmentService', '''<f0> AppointmentService|<f1> |<f2> + createAppointment()\l''')
        service.node('PatientService', '''<f0> PatientService|<f1> |<f2> + getPatientDetails()\l''')
    
    with diagram.subgraph(name='cluster_2') as repository:
        repository.attr(label='Repository Layer', style='filled', fillcolor='#E9F7EF')
        repository.node('AppointmentRepository', '''<f0> AppointmentRepository|<f1> |<f2> + save()\l''')
        repository.node('PatientRepository', '''<f0> PatientRepository|<f1> |<f2> + save()\l''')
    
    with diagram.subgraph(name='cluster_3') as domain:
        domain.attr(label='Domain Layer', style='filled', fillcolor='#F4ECF7')
        domain.node('Appointment', '''<f0> Appointment|<f1> |<f2> + setStatus()\l''')
        domain.node('Patient', '''<f0> Patient|<f1> |<f2> + getId()\l''')
    
    # Add inter-layer relationships
    diagram.edge('AppointmentView', 'AppointmentService', 'uses')
    diagram.edge('AppointmentService', 'AppointmentRepository', 'uses')
    diagram.edge('AppointmentRepository', 'Appointment', 'manages')
    
    return diagram

def generate_all_diagrams():
    # Generate individual layer diagrams
    presentation = create_presentation_layer()
    presentation.render(cleanup=True)
    
    service = create_service_layer()
    service.render(cleanup=True)
    
    repository = create_repository_layer()
    repository.render(cleanup=True)
    
    domain = create_domain_layer()
    domain.render(cleanup=True)
    
    # Generate combined diagram
    combined = create_combined_diagram()
    combined.render(cleanup=True)

if __name__ == "__main__":
    generate_all_diagrams()
